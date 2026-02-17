"""Main BaSiC class."""

# Core modules
from __future__ import annotations

import json
import os
import time
import random
from enum import Enum
from multiprocessing import cpu_count
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import jax.numpy as jnp
import numpy as np
from jax import device_put
from jax.image import ResizeMethod
from jax.image import resize as jax_resize
from pydantic import BaseModel, Field, PrivateAttr, model_validator #root_validator, 
from skimage.filters import threshold_otsu
from skimage.morphology import ball, binary_erosion
from skimage.transform import resize as skimage_resize

from image_profiler.preprocessing.basic.dct_tools import JaxDCT
from image_profiler.preprocessing.basic.jax_routines import ApproximateFit, LadmapFit
from image_profiler.preprocessing.basic.metrics import autotune_cost

ArrayLike = Union[np.ndarray, jnp.ndarray]  # dask.array.Array, zarr.Array
PathLike = Union[str, Path]
newax = jnp.newaxis

# Get number of available threads to limit CPU thrashing
# From preadator: https://pypi.org/project/preadator/
if hasattr(os, "sched_getaffinity"):
    # On Linux, we can detect how many cores are assigned to this process.
    # This is especially useful when running in a Docker container, when the
    # number of cores is intentionally limited.
    NUM_THREADS = len(os.sched_getaffinity(0))  # type: ignore
else:
    # Default back to multiprocessing cpu_count, which is always going to count
    # the total number of cpus
    NUM_THREADS = cpu_count()


class HillClimbingOptimizer:
    """Hill Climbing Optimizer - Replacement for hyperactive.optimizers.HillClimbingOptimizer.
    
    A simple hill climbing optimization algorithm that explores the search space
    by generating neighbours around the current best solution.
    """
    
    def __init__(
        self,
        epsilon: float = 0.1,
        distribution: str = "laplace",
        n_neighbours: int = 4,
        rand_rest_p: float = 0.1,
    ):
        """Initialize the Hill Climbing Optimizer.
        
        Args:
            epsilon: Step size for neighbourhood exploration.
            distribution: Distribution for generating neighbours ('laplace' or 'gaussian').
            n_neighbours: Number of neighbours to generate per iteration.
            rand_rest_p: Probability of random restart to escape local optima.
        """
        self.epsilon = epsilon
        self.distribution = distribution
        self.n_neighbours = n_neighbours
        self.rand_rest_p = rand_rest_p
    
    def suggest(self, search_space: Dict, current_best: Optional[Dict] = None, random_state: Optional[int] = None) -> Dict:
        """Suggest a new parameter configuration.
        
        Args:
            search_space: Dictionary defining the search space for each parameter.
            current_best: Current best parameters found so far.
            random_state: Random seed for reproducibility.
            
        Returns:
            A new parameter configuration to evaluate.
        """
        if random_state is not None:
            random.seed(random_state)
        
        # If no current best, generate initial random parameters
        if current_best is None:
            return self._generate_random_params(search_space)
        
        # Random restart with probability rand_rest_p
        if random.random() < self.rand_rest_p:
            return self._generate_random_params(search_space)
        
        # Generate neighbours around current best
        return self._generate_neighbour(search_space, current_best)
    
    def _generate_random_params(self, search_space: Dict) -> Dict:
        """Generate random parameters from the search space."""
        params = {}
        for key, values in search_space.items():
            params[key] = random.choice(values)
        return params
    
    def _generate_neighbour(self, search_space: Dict, current: Dict) -> Dict:
        """Generate a neighbour solution around the current best."""
        neighbour = current.copy()
        
        # Select a random parameter to modify
        param_to_modify = random.choice(list(search_space.keys()))
        values = search_space[param_to_modify]
        
        if param_to_modify not in current:
            neighbour[param_to_modify] = random.choice(values)
            return neighbour
        
        current_value = current[param_to_modify]
        current_idx = values.index(current_value) if current_value in values else 0
        
        # Generate neighbours by moving to adjacent values in the search space
        neighbours_idx = []
        for delta in range(-self.n_neighbours // 2, self.n_neighbours // 2 + 1):
            if delta == 0:
                continue
            new_idx = current_idx + delta
            if 0 <= new_idx < len(values):
                neighbours_idx.append(new_idx)
        
        if neighbours_idx:
            # Select a neighbour based on distribution
            if self.distribution == "laplace":
                # Laplace distribution - prefer closer neighbours
                weights = [1.0 / (abs(current_idx - idx) + 1) for idx in neighbours_idx]
            else:
                # Gaussian - all neighbours equally likely
                weights = [1.0] * len(neighbours_idx)
            
            total_weight = sum(weights)
            weights = [w / total_weight for w in weights]
            selected_idx = random.choices(neighbours_idx, weights=weights, k=1)[0]
            neighbour[param_to_modify] = values[selected_idx]
        else:
            # If no valid neighbours, generate random
            neighbour[param_to_modify] = random.choice(values)
        
        return neighbour


class Hyperactive:
    """Hyperactive - Replacement for hyperactive.Hyperactive.
    
    A simple optimization wrapper that performs hill climbing optimization
    over a given search space and objective function.
    """
    
    def __init__(self):
        """Initialize Hyperactive."""
        self.searches = []
        self.results = {}
        self.best_params = {}
        self.best_scores = {}
    
    def add_search(self, objective_func, search_space: Dict, **kwargs):
        """Add a search to the optimizer.
        
        Args:
            objective_func: The objective function to maximize.
            search_space: Dictionary defining the search space.
            **kwargs: Additional configuration options including:
                - optimizer: Optimizer instance to use
                - n_iter: Number of iterations
                - initialize: Initialization options (warm_start)
                - random_state: Random seed
                - early_stopping: Early stopping configuration
        """
        search_config = {
            'objective_func': objective_func,
            'search_space': search_space,
            'optimizer': kwargs.get('optimizer', HillClimbingOptimizer()),
            'n_iter': kwargs.get('n_iter', 100),
            'init_params': kwargs.get('initialize', {}).get('warm_start', [{}])[0] if kwargs.get('initialize', {}).get('warm_start') else {},
            'random_state': kwargs.get('random_state'),
            'early_stopping': kwargs.get('early_stopping', None),
        }
        self.searches.append(search_config)
    
    def run(self):
        """Run all added searches."""
        for search in self.searches:
            self._run_search(search)
    
    def _run_search(self, search: Dict):
        """Run a single search optimization.
        
        Args:
            search: Search configuration dictionary.
        """
        objective_func = search['objective_func']
        search_space = search['search_space']
        optimizer = search['optimizer']
        n_iter = search['n_iter']
        init_params = search['init_params']
        random_state = search['random_state']
        early_stopping = search['early_stopping']
        
        if random_state is not None:
            random.seed(random_state)
        
        # Initialize with warm start if provided
        current_best = init_params.copy() if init_params else None
        best_score = float('-inf')
        iteration = 0
        no_change_count = 0
        
        # Evaluate initial parameters if provided
        if current_best:
            score = objective_func(current_best)
            if score > best_score:
                best_score = score
                current_best = current_best.copy()
            else:
                current_best = None
        
        # Main optimization loop
        for i in range(n_iter):
            iteration = i
            
            # Get next suggestion from optimizer
            suggested_params = optimizer.suggest(search_space, current_best, random_state)
            
            # Evaluate objective function
            score = objective_func(suggested_params)
            
            # Update best if improved
            if score > best_score:
                best_score = score
                current_best = suggested_params.copy()
                no_change_count = 0
            else:
                no_change_count += 1
            
            # Check early stopping
            if early_stopping and no_change_count >= early_stopping.get('n_iter_no_change', 10):
                break
        
        # Store results
        search_key = str(objective_func)
        self.results[search_key] = {
            'best_params': current_best,
            'best_score': best_score,
            'n_iterations': iteration + 1,
        }
        self.best_params[search_key] = current_best
        self.best_scores[search_key] = best_score
    
    def best_para(self, objective_func) -> Dict:
        """Get the best parameters found for a given objective function.
        
        Args:
            objective_func: The objective function to get best params for.
            
        Returns:
            Dictionary of best parameters found.
        """
        search_key = str(objective_func)
        return self.best_params.get(search_key, {})


class FittingMode(str, Enum):
    """Fit method enum."""

    ladmap: str = "ladmap"
    approximate: str = "approximate"


class ResizeMode(str, Enum):
    """Resize method enum."""

    jax: str = "jax"
    skimage: str = "skimage"
    skimage_dask: str = "skimage_dask"


class TimelapseTransformMode(str, Enum):
    """Timelapse transformation enum."""

    additive: str = "additive"
    multiplicative: str = "multiplicative"


_SETTINGS_FNAME = "settings.json"
_PROFILES_FNAME = "profiles.npz"


# multiple channels should be handled by creating a `basic` object for each channel
class BaSiC(BaseModel):
    """A class for fitting and applying BaSiC illumination correction profiles."""

    baseline: Optional[np.ndarray] = Field(
        None,
        description="Holds the baseline for the shading model.",
        exclude=True,  # Don't dump to output json/yaml
    )
    darkfield: np.ndarray = Field(
        default_factory=lambda: np.zeros((128, 128), dtype=np.float64),
        description="Holds the darkfield component for the shading model.",
        exclude=True,  # Don't dump to output json/yaml
    )
    fitting_mode: FittingMode = Field(
        FittingMode.ladmap, description="Must be one of ['ladmap', 'approximate']"
    )
    epsilon: float = Field(
        0.1,
        description="Weight regularization term.",
    )
    flatfield: np.ndarray = Field(
        default_factory=lambda: np.zeros((128, 128), dtype=np.float64),
        description="Holds the flatfield component for the shading model.",
        exclude=True,  # Don't dump to output json/yaml
    )
    get_darkfield: bool = Field(
        False,
        description="When True, will estimate the darkfield shading component.",
    )
    smoothness_flatfield: float = Field(
        1.0, description="Weight of the flatfield term in the Lagrangian."
    )
    smoothness_darkfield: float = Field(
        1.0, description="Weight of the darkfield term in the Lagrangian."
    )
    sparse_cost_darkfield: float = Field(
        0.01, description="Weight of the darkfield sparse term in the Lagrangian."
    )
    autosegment: bool = Field(
        False,
        description="When not False, automatically segment the image before fitting."
        "When True, `threshold_otsu` from `scikit-image` is used "
        "and the brighter pixels are taken."
        "When a callable is given, it is used as the segmentation function.",
    )
    autosegment_margin: int = Field(
        10,
        description="Margin of the segmentation mask to the thresholded region.",
    )
    max_iterations: int = Field(
        500,
        description="Maximum number of iterations for single optimization.",
    )
    max_reweight_iterations: int = Field(
        10,
        description="Maximum number of reweighting iterations.",
    )
    max_reweight_iterations_baseline: int = Field(
        5,
        description="Maximum number of reweighting iterations for baseline.",
    )
    max_workers: int = Field(
        NUM_THREADS,
        description="Maximum number of threads used for processing.",
        exclude=True,  # Don't dump to output json/yaml
    )
    rho: float = Field(1.5, description="Parameter rho for mu update.")
    mu_coef: float = Field(12.5, description="Coefficient for initial mu value.")
    max_mu_coef: float = Field(
        1e7, description="Maximum allowed value of mu, divided by the initial value."
    )
    optimization_tol: float = Field(
        1e-3,
        description="Optimization tolerance.",
    )
    optimization_tol_diff: float = Field(
        1e-2,
        description="Optimization tolerance for update diff.",
    )
    resize_mode: ResizeMode = Field(
        ResizeMode.jax,
        description="Resize mode to use when downsampling images. "
        + "Must be one of 'jax', 'skimage', and 'skimage_dask'",
    )
    resize_params: Dict = Field(
        {},
        description="Parameters for the resize function when downsampling images.",
    )
    reweighting_tol: float = Field(
        1e-2,
        description="Reweighting tolerance in mean absolute difference of images.",
    )
    sort_intensity: bool = Field(
        False,
        description="Whether or not to sort the intensities of the image.",
    )
    working_size: Optional[Union[int, List[int]]] = Field(
        128,
        description="Size for running computations. None means no rescaling.",
    )

    # Private attributes for internal processing
    _score: float = PrivateAttr(None)
    _reweight_score: float = PrivateAttr(None)
    _weight: float = PrivateAttr(None)
    _weight_dark: float = PrivateAttr(None)
    _residual: float = PrivateAttr(None)
    _S: float = PrivateAttr(None)
    _B: float = PrivateAttr(None)
    _D_R: float = PrivateAttr(None)
    _D_Z: float = PrivateAttr(None)
    _smoothness_flatfield: float = PrivateAttr(None)
    _smoothness_darkfield: float = PrivateAttr(None)
    _sparse_cost_darkfield: float = PrivateAttr(None)

    class Config:
        """Pydantic class configuration."""

        arbitrary_types_allowed = True
        extra = "forbid"

    # @root_validator(pre=True)
    @model_validator(mode='before')
    def debug_log_values(cls, values: Dict[str, Any]):
        """Use a validator to echo input values."""
        return values

    def __call__(
        self, images: np.ndarray, timelapse: bool = False
    ) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        """Shortcut for `BaSiC.transform`."""
        return self.transform(images, timelapse)

    def _resize(self, Im, target_shape):
        if self.resize_mode == ResizeMode.jax:
            resize_params = dict(method=ResizeMethod.LINEAR)
            resize_params.update(self.resize_params)
            Im = device_put(Im).astype(jnp.float32)
            return jax_resize(Im, target_shape, **resize_params)

        elif self.resize_mode == ResizeMode.skimage:
            Im = skimage_resize(
                np.array(Im), target_shape, preserve_range=True, **self.resize_params
            )
            return device_put(Im).astype(jnp.float32)

        elif self.resize_mode == ResizeMode.skimage_dask:
            assert np.array_equal(target_shape[:-2], Im.shape[:-2])
            import dask.array as da

            Im = (
                da.from_array(
                    [
                        skimage_resize(
                            np.array(Im[tuple(inds)]),
                            target_shape[-2:],
                            preserve_range=True,
                            **self.resize_params,
                        )
                        for inds in np.ndindex(Im.shape[:-2])
                    ]
                )
                .reshape((*Im.shape[:-2], *target_shape[-2:]))
                .compute()
            )
            return device_put(Im).astype(jnp.float32)

    def _resize_to_working_size(self, Im):
        """Resize the images to the working size."""
        if self.working_size is not None:
            if np.isscalar(self.working_size):
                working_shape = [self.working_size] * (Im.ndim - 2)
            else:
                if not Im.ndim - 2 == len(self.working_size):
                    raise ValueError(
                        "working_size must be a scalar or match the image dimensions"
                    )
                else:
                    working_shape = self.working_size
            target_shape = [*Im.shape[:2], *working_shape]
            Im = self._resize(Im, target_shape)

        return Im

    def _perform_segmentation(self, Im):
        """Perform segmentation on the images."""
        if not self.autosegment:
            return np.ones_like(Im, dtype=bool)
        elif self.autosegment is True:
            th = threshold_otsu(Im)
            mask = Im < th
            return np.array(
                [binary_erosion(m, ball(self.autosegment_margin)) for m in mask]
            )
        else:
            return self.autosegment(Im)

    def fit(
        self,
        images: np.ndarray,
        fitting_weight: Optional[np.ndarray] = None,
        skip_shape_warning=False,
    ) -> None:
        """Generate illumination correction profiles from images.

        Args:
            images: Input images to fit shading model.
                    Must be 3-dimensional or 4-dimensional array
                    with dimension of (T,Y,X) or (T,Z,Y,X).
                    T can be either of time or mosaic position.
                    Multichannel images should be
                    independently corrected for each channel.
            fitting_weight: Relative fitting weight for each pixel.
                    Higher value means more contribution to fitting.
                    Must has the same shape as images.
            skip_shape_warning: if True, warning for last dimension
                    less than 10 is suppressed.

        Example:
            >>> from basicpy import BaSiC
            >>> from basicpy import datasets as bdata
            >>> images = bdata.wsi_brain()
            >>> basic = BaSiC()  # use default settings
            >>> basic.fit(images)

        """
        ndim = images.ndim
        if images.ndim == 3:
            images = images[:, np.newaxis, ...]
            if fitting_weight is not None:
                fitting_weight = fitting_weight[:, np.newaxis, ...]
        elif images.ndim == 4:
            if self.fitting_mode == FittingMode.approximate:
                raise ValueError(
                    "Only 2-dimensional images are accepted for the approximate mode."
                )
        else:
            raise ValueError(
                "Images must be 3 or 4-dimensional array, "
                + "with dimension of (T,Y,X) or (T,Z,Y,X)."
            )

        if images.shape[-1] < 10 and not skip_shape_warning:
            pass

        if fitting_weight is not None and fitting_weight.shape != images.shape:
            raise ValueError("fitting_weight must have the same shape as images.")

        start_time = time.monotonic()

        Im = self._resize_to_working_size(images)

        if fitting_weight is not None:
            Ws = device_put(fitting_weight).astype(jnp.float32)
            Ws = self._resize_to_working_size(Ws)
            # normalize relative weight to 0 to 1
            Ws_min = jnp.min(Ws)
            Ws_max = jnp.max(Ws)
            Ws = (Ws - Ws_min) / (Ws_max - Ws_min)
        else:
            Ws = jnp.ones_like(Im)

        Ws = Ws * self._perform_segmentation(Im)

        # Im2 and Ws2 will possibly be sorted
        if self.sort_intensity:
            inds = jnp.argsort(Im, axis=0)
            Im2 = jnp.take_along_axis(Im, inds, axis=0)
            Ws2 = jnp.take_along_axis(Ws, inds, axis=0)
        else:
            Im2 = Im
            Ws2 = Ws

        if self.fitting_mode == FittingMode.approximate:
            mean_image = jnp.mean(Im2, axis=0)
            mean_image = mean_image / jnp.mean(Im2)
            mean_image_dct = JaxDCT.dct3d(mean_image.T)
            self._smoothness_flatfield = (
                jnp.sum(jnp.abs(mean_image_dct)) / 800 * self.smoothness_flatfield
            )
            self._smoothness_darkfield = (
                self._smoothness_flatfield * self.smoothness_darkfield / 2.5
            )
            self._sparse_cost_darkfield = (
                self._smoothness_darkfield * self.sparse_cost_darkfield * 100
            )
        else:
            self._smoothness_flatfield = self.smoothness_flatfield
            self._smoothness_darkfield = self.smoothness_darkfield
            self._sparse_cost_darkfield = self.sparse_cost_darkfield

        # spectral_norm = jnp.linalg.norm(Im.reshape((Im.shape[0], -1)), ord=2)
        _temp = jnp.linalg.svd(Im2.reshape((Im2.shape[0], -1)), full_matrices=False)
        spectral_norm = _temp[1][0]

        if self.fitting_mode == FittingMode.approximate:
            init_mu = self.mu_coef / spectral_norm
        else:
            init_mu = self.mu_coef / spectral_norm / np.prod(Im2.shape)
        fit_params = self.dict()
        fit_params.update(
            dict(
                smoothness_flatfield=self._smoothness_flatfield,
                smoothness_darkfield=self._smoothness_darkfield,
                sparse_cost_darkfield=self._sparse_cost_darkfield,
                # matrix 2-norm (largest sing. value)
                init_mu=init_mu,
                max_mu=init_mu * self.max_mu_coef,
                D_Z_max=jnp.min(Im2),
                image_norm=jnp.linalg.norm(Im2.flatten(), ord=2),
            )
        )

        # Initialize variables
        W = jnp.ones_like(Im2, dtype=jnp.float32) * Ws2
        W_D = jnp.ones(Im2.shape[1:], dtype=jnp.float32)
        last_S = None
        last_D = None
        S = None
        D = None
        B = None

        if self.fitting_mode == FittingMode.ladmap:
            fitting_step = LadmapFit(**fit_params)
        else:
            fitting_step = ApproximateFit(**fit_params)

        for i in range(self.max_reweight_iterations):
            if self.fitting_mode == FittingMode.approximate:
                S = jnp.zeros(Im2.shape[1:], dtype=jnp.float32)
            else:
                S = jnp.median(Im2, axis=0)
            D_R = jnp.zeros(Im2.shape[1:], dtype=jnp.float32)
            D_Z = 0.0
            if self.fitting_mode == FittingMode.approximate:
                B = jnp.ones(Im2.shape[0], dtype=jnp.float32)
            else:
                B = jnp.ones(Im2.shape[0], dtype=jnp.float32)
            I_R = jnp.zeros(Im2.shape, dtype=jnp.float32)
            S, D_R, D_Z, I_R, B, norm_ratio, converged = fitting_step.fit(
                Im2,
                W,
                W_D,
                S,
                D_R,
                D_Z,
                B,
                I_R,
            )
            self._score = norm_ratio
            if not converged:
                pass
            if S.max() == 0:
                raise RuntimeError(
                    "Estimated flatfield is zero. "
                    + "Please try to decrease smoothness_darkfield."
                )
            self._S = S
            self._D_R = D_R
            self._B = B
            self._D_Z = D_Z
            D = fitting_step.calc_darkfield(S, D_R, D_Z)  # darkfield
            mean_S = jnp.mean(S)
            S = S / mean_S  # flatfields
            B = B * mean_S  # baseline
            I_B = B[:, newax, newax, newax] * S[newax, ...] + D[newax, ...]
            W = fitting_step.calc_weights(I_B, I_R) * Ws2
            W_D = fitting_step.calc_dark_weights(D_R)

            self._weight = W
            self._weight_dark = W_D
            self._residual = I_R

            if last_S is not None:
                mad_flatfield = jnp.sum(jnp.abs(S - last_S)) / jnp.sum(np.abs(last_S))
                if self.get_darkfield:
                    mad_darkfield = jnp.sum(jnp.abs(D - last_D)) / max(
                        jnp.sum(jnp.abs(last_D)), 1
                    )  # assumes the amplitude of darkfield is more than 1
                    self._reweight_score = max(mad_flatfield, mad_darkfield)
                else:
                    self._reweight_score = mad_flatfield

                if self._reweight_score <= self.reweighting_tol:
                    break
            if i == self.max_reweight_iterations - 1:
                pass
            last_S = S
            last_D = D

        if not converged:
            pass

        assert S is not None
        assert D is not None
        assert B is not None

        if self.sort_intensity:
            for i in range(self.max_reweight_iterations_baseline):
                B = jnp.ones(Im.shape[0], dtype=jnp.float32)
                if self.fitting_mode == FittingMode.approximate:
                    B = jnp.mean(Im, axis=(1, 2, 3))
                I_R = jnp.zeros(Im.shape, dtype=jnp.float32)
                I_R, B, norm_ratio, converged = fitting_step.fit_baseline(
                    Im,
                    W,
                    S,
                    D,
                    B,
                    I_R,
                )

                I_B = B[:, newax, newax, newax] * S[newax, ...] + D[newax, ...]
                W = fitting_step.calc_weights_baseline(I_B, I_R) * Ws
                self._weight = W
                self._residual = I_R

        self.flatfield = skimage_resize(S, images.shape[1:])
        self.darkfield = skimage_resize(D, images.shape[1:])
        if ndim == 3:
            self.flatfield = self.flatfield[0]
            self.darkfield = self.darkfield[0]
        self.baseline = B

    def transform(
        self,
        images: np.ndarray,
        timelapse: Union[bool, TimelapseTransformMode] = False,
        frames: Optional[Sequence[Union[int, np.int_]]] = None,
    ) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        """Apply profile to images.

        Args:
            images: input images to correct. See `fit`.
            timelapse: If `True`, corrects the timelapse/photobleaching offsets,
                       assuming that the residual is the product of flatfield and
                       the object fluorescence. Also accepts "multiplicative"
                       (the same as `True`) or "additive" (residual is the object
                       fluorescence).
            frames: Frames to use for transformation. Defaults to None (all frames).

        Returns:
            corrected images

        Example:
            >>> basic.fit(images)
            >>> corrected = basic.transform(images)
        """
        if self.baseline is None:
            raise RuntimeError("BaSiC object is not initialized")

        start_time = time.monotonic()

        # Convert to the correct format
        im_float = images.astype(np.float32)

        # Image = B_n x S_l + D_l + I_R_nl

        # in timelapse cases ...
        # "Multiplicative" mode
        # Real Image x S_l = I_R_nl
        # Image = (B_n + Real Image) x S_l + D_l
        # Real Image = (Image - D_l) / S_l - B_n

        # "Additive" mode
        # Real Image = I_R_nl
        # Image = B_n x S_l + D_l + Real Image
        # Real Image = Image - D_l - (S_l x B_n)

        # in non-timelapse cases ...
        # we assume B_n is the mean of Real Image
        # and then always assume Multiplicative mode
        # the image model is
        # Image = Real Image x S_l + D_l
        # Real Image = (Image - D_l) / S_l

        if timelapse:
            if timelapse is True:
                timelapse = TimelapseTransformMode.multiplicative
            if frames is None:
                _frames = slice(None)
            else:
                _frames = np.array(frames)
            baseline_inds = tuple([_frames] + ([np.newaxis] * (im_float.ndim - 1)))
            if timelapse == TimelapseTransformMode.multiplicative:
                output = (im_float - self.darkfield[np.newaxis]) / self.flatfield[
                    np.newaxis
                ] - self.baseline[baseline_inds]
            elif timelapse == TimelapseTransformMode.additive:
                baseline_flatfield = (
                    self.flatfield[np.newaxis] * self.baseline[baseline_inds]
                )
                output = im_float - self.darkfield[np.newaxis] - baseline_flatfield
            else:
                raise ValueError(
                    "timelapse value must be bool, 'multiplicative' or 'additive'"
                )
        else:
            output = (im_float - self.darkfield[np.newaxis]) / self.flatfield[
                np.newaxis
            ]
        return output

    # REFACTOR large datasets will probably prefer saving corrected images to
    # files directly, a generator may be handy
    def fit_transform(
        self,
        images: ArrayLike,
        fitting_weight: Optional[np.ndarray] = None,
        skip_shape_warning=False,
        timelapse: bool = False,
    ) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        """Fit and transform on data.

        Args:
            images: input images to fit and correct. See `fit`.

        Returns:
            corrected images

        Example:
            >>> corrected = basic.fit_transform(images)
        """
        self.fit(
            images, fitting_weight=fitting_weight, skip_shape_warning=skip_shape_warning
        )
        corrected = self.transform(images, timelapse)

        return corrected

    def autotune(
        self,
        images: np.ndarray,
        fitting_weight: Optional[np.ndarray] = None,
        skip_shape_warning: bool = False,
        *,
        optmizer=None,
        n_iter=100,
        search_space=None,
        init_params=None,
        timelapse: bool = False,
        histogram_qmin: float = 0.01,
        histogram_qmax: float = 0.99,
        vmin_factor: float = 0.6,
        vrange_factor: float = 1.5,
        histogram_bins: int = 1000,
        histogram_use_fitting_weight: bool = True,
        fourier_l0_norm_image_threshold: float = 0.1,
        fourier_l0_norm_fourier_radius=10,
        fourier_l0_norm_threshold=0.0,
        fourier_l0_norm_cost_coef=30,
        early_stop: bool = True,
        early_stop_n_iter_no_change: int = 15,
        early_stop_torelance: float = 1e-6,
        random_state: Optional[int] = None,
    ) -> None:
        """Automatically tune the parameters of the model.

        Args:
            images: input images to fit and correct. See `fit`.
            fitting_weight: Relative fitting weight for each pixel. See `fit`.
            skip_shape_warning: if True, warning for last dimension
                    less than 10 is suppressed.
            optimizer: optimizer to use. Defaults to
                    `HillClimbingOptimizer` (our custom implementation).
            n_iter: number of iterations for the optimizer. Defaults to 100.
            search_space: search space for the optimizer.
                    Defaults to a reasonable range for each parameter.
            init_params: initial parameters for the optimizer.
                    Defaults to a reasonable initial value for each parameter.
            timelapse: if True, corrects the timelapse/photobleaching offsets.
            histogram_qmin: the minimum quantile to use for the histogram.
                    Defaults to 0.01.
            histogram_qmax: the maximum quantile to use for the histogram.
                    Defaults to 0.99.
            histogram_bins: the number of bins to use for the histogram.
                    Defaults to 100.
            hisogram_use_fitting_weight: if True, uses the weight for the histogram.
                    Defaults to True.
            fourier_l0_norm_image_threshold : float
                The threshold for image values for the fourier L0 norm calculation.
            fourier_l0_norm_fourier_radius : float
                The Fourier radius for the fourier L0 norm calculation.
            fourier_l0_norm_threshold : float
                The maximum preferred value for the fourier L0 norm.
            fourier_l0_norm_cost_coef : float
                The cost coefficient for the fourier L0 norm.
            early_stop: if True, stops the optimization when the change in
                    entropy is less than `early_stop_torelance`.
                    Defaults to True.
            early_stop_n_iter_no_change: the number of iterations for early
                    stopping. Defaults to 10.
            early_stop_torelance: the absolute value torelance
                    for early stopping.
            random_state: random state for the optimizer.

        """

        if search_space is None:
            search_space = {
                "smoothness_flatfield": list(np.logspace(-3, 1, 15)),
            }
            if self.get_darkfield:
                search_space.update(
                    {
                        "smoothness_darkfield": [0] + list(np.logspace(-3, 1, 15)),
                        "sparse_cost_darkfield": [0] + list(np.logspace(-3, 1, 15)),
                    }
                )
        if init_params is None:
            init_params = {
                "smoothness_flatfield": 0.1,
            }
            if self.get_darkfield:
                init_params.update(
                    {
                        "smoothness_darkfield": 1e-3,
                        "sparse_cost_darkfield": 1e-3,
                    }
                )

        # calculate the histogram range
        basic = self.copy(update=init_params)
        basic.fit(
            images,
            fitting_weight=fitting_weight,
            skip_shape_warning=skip_shape_warning,
        )
        transformed = basic.transform(images, timelapse=timelapse)
        vmin, vmax = np.quantile(transformed, [histogram_qmin, histogram_qmax])
        val_range = (
            vmax - vmin * vmin_factor
        ) * vrange_factor  # fix the value range for histogram

        if fitting_weight is None or not histogram_use_fitting_weight:
            weights = None
        else:
            weights = fitting_weight

        def fit_and_calc_entropy(params):
            try:
                basic = self.copy(update=params)
                basic.fit(
                    images,
                    fitting_weight=fitting_weight,
                    skip_shape_warning=skip_shape_warning,
                )
                transformed = basic.transform(images, timelapse=timelapse)
                vmin_new = np.quantile(transformed, histogram_qmin) * vmin_factor

                if np.allclose(basic.flatfield, np.ones_like(basic.flatfield)):
                    return -np.inf  # discard the case where flatfield is all ones

                return -1.0 * autotune_cost(
                    transformed,
                    basic.flatfield,
                    entropy_vmin=vmin_new,
                    entropy_vmax=vmin_new + val_range,
                    histogram_bins=histogram_bins,
                    fourier_l0_norm_cost_coef=fourier_l0_norm_cost_coef,
                    fourier_l0_norm_image_threshold=fourier_l0_norm_image_threshold,
                    fourier_l0_norm_fourier_radius=fourier_l0_norm_fourier_radius,
                    fourier_l0_norm_threshold=fourier_l0_norm_threshold,
                    weights=weights,
                )
            except RuntimeError:
                return -np.inf

        if optmizer is None:
            optimizer = HillClimbingOptimizer(
                epsilon=0.1,
                distribution="laplace",
                n_neighbours=4,
                rand_rest_p=0.1,
            )

        hyper = Hyperactive()

        params = dict(
            optimizer=optimizer,
            n_iter=n_iter,
            initialize=dict(warm_start=[init_params]),
            random_state=random_state,
        )

        if early_stop:
            params.update(
                dict(
                    early_stopping=dict(
                        n_iter_no_change=early_stop_n_iter_no_change,
                        tol_abs=early_stop_torelance,
                    )
                )
            )

        hyper.add_search(
            fit_and_calc_entropy,
            search_space,
            **params,
        )
        hyper.run()
        best_params = hyper.best_para(fit_and_calc_entropy)
        self.__dict__.update(best_params)

    @property
    def score(self):
        """The BaSiC fit final score."""
        return self._score

    @property
    def reweight_score(self):
        """The BaSiC fit final reweighting score."""
        return self._reweight_score

    @property
    def settings(self) -> Dict:
        """Current settings.

        Returns:
            current settings
        """
        return self.dict()

    def save_model(self, model_dir: PathLike, overwrite: bool = False) -> None:
        """Save current model to folder.

        Args:
            model_dir: path to model directory

        Raises:
            FileExistsError: if model directory already exists
        """
        path = Path(model_dir)

        try:
            path.mkdir()
        except FileExistsError:
            if not overwrite:
                raise FileExistsError("Model folder already exists.")

        # save settings
        with open(path / _SETTINGS_FNAME, "w") as fp:
            # see pydantic docs for output options
            fp.write(self.json())

        # NOTE emit warning if profiles are all zeros? fit probably not run
        # save profiles
        np.savez(
            path / _PROFILES_FNAME,
            flatfield=np.array(self.flatfield),
            darkfield=np.array(self.darkfield),
            baseline=np.array(self.baseline),
        )

    @classmethod
    def load_model(cls, model_dir: PathLike) -> BaSiC:
        """Create a new instance from a model folder."""
        path = Path(model_dir)

        if not path.exists():
            raise FileNotFoundError("Model directory not found.")

        with open(path / _SETTINGS_FNAME) as fp:
            model = json.load(fp)

        profiles = np.load(path / _PROFILES_FNAME)
        model["flatfield"] = profiles["flatfield"]
        model["darkfield"] = profiles["darkfield"]
        model["baseline"] = profiles["baseline"]

        return BaSiC(**model)

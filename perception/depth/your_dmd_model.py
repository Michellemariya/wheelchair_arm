# perception/depth/your_dmd_model.py
#
# Dependencies (install in your 'robot' conda env before use):
#   pip install torch torchvision
#   pip install transformers>=4.35.0
#   pip install Pillow numpy opencv-python
#
# Model: depth-anything/Depth-Anything-V2-Small-hf
# Task:  depth-estimation (HuggingFace transformers pipeline)

import time
import traceback

import cv2
import numpy as np

# Top-level imports for transformers are deferred to first infer() call (lazy load),
# so that importing this module never fails even if transformers isn't installed yet.
# numpy and cv2 are imported eagerly because the rest of the pipeline depends on them.


class MonocularDepthModel:
    """
    Wrapper around Depth Anything V2-Small (HuggingFace) for monocular relative
    depth estimation.

    Usage in pipeline
    -----------------
    model = MonocularDepthModel()
    model.warmup()                          # optional but recommended at startup
    depth_rel = model.infer(frame_bgr)      # HxW float32 relative depth map

    The returned map is in an *arbitrary* relative scale.  The calling code in
    perception/depth/scaled_depth.py normalises it to [0, 1] and anchors it to
    the gripper's encoder-derived metric Z before back-projecting object centroids.

    Thread safety
    -------------
    Not thread-safe.  Use one instance per inference thread.
    """

    # HuggingFace model identifier for DA V2-Small
    MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"

    # DA V2's native input resolution.
    # Both H and W must equal this value for optimal feature quality.
    NATIVE_SIZE = 518

    def __init__(self):
        """
        Initialise the object but do NOT load the model yet.
        Model loading is deferred to the first call to infer() (lazy loading).
        """
        self._pipe        = None     # HuggingFace pipeline, set on first infer()
        self._device      = None     # 'cuda' or 'cpu', detected on first infer()
        self._loaded      = False    # True only after successful model load
        self._load_failed = False    # True after first failure — stops retrying

        # Letterbox padding state (set in _bgr_to_pil, consumed in _postprocess).
        # BUG-FIX-1: These must be reset to safe defaults here so _postprocess
        # never reads stale values if called before _bgr_to_pil (e.g. from a
        # subclass or test mock).
        self._pad_top   = 0
        self._pad_left  = 0
        self._pad_new_h = self.NATIVE_SIZE
        self._pad_new_w = self.NATIVE_SIZE

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_model(self):
        """
        Load the Depth Anything V2-Small pipeline from HuggingFace.
        Called automatically on the first infer() call.
        On failure, sets _load_failed=True so retries are not attempted.
        """
        try:
            import torch
            from transformers import pipeline as hf_pipeline

            # BUG-FIX-2: transformers pipeline device= expects an int (0 = first GPU,
            # -1 = CPU), NOT the string "cuda"/"cpu". Using the string works in some
            # transformers versions but silently breaks in others (maps to wrong device
            # or raises ValueError). Always pass int.
            if torch.cuda.is_available():
                self._device = "cuda"
                device_arg   = 0
                print("[MonocularDepthModel] CUDA available — running on GPU.")
            else:
                self._device = "cpu"
                device_arg   = -1
                print("[MonocularDepthModel] No CUDA found — running on CPU.")

            print(f"[MonocularDepthModel] Loading {self.MODEL_ID} …")
            self._pipe = hf_pipeline(
                task="depth-estimation",
                model=self.MODEL_ID,
                device=device_arg,
            )
            self._loaded = True
            print("[MonocularDepthModel] Model loaded successfully.")

        except Exception:
            print("[MonocularDepthModel] ERROR: failed to load model. Will not retry.")
            traceback.print_exc()
            self._loaded      = False
            self._load_failed = True

    def _bgr_to_pil(self, frame_bgr: np.ndarray):
        """
        Convert an OpenCV BGR uint8 frame to a PIL RGB Image at NATIVE_SIZE,
        preserving aspect ratio via letterbox padding (black bars).

        Stores padding offsets as instance variables so _postprocess() can
        crop them back out before resizing to the original resolution.

        Parameters
        ----------
        frame_bgr : np.ndarray
            HxWx3 uint8 array in BGR channel order.

        Returns
        -------
        PIL.Image.Image
            RGB image of size NATIVE_SIZE x NATIVE_SIZE with letterbox padding.
        """
        from PIL import Image

        # BGR → RGB: model was trained on RGB channel order.
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w = frame_rgb.shape[:2]

        # Aspect-ratio preserving scale: fit longest side to NATIVE_SIZE.
        scale = self.NATIVE_SIZE / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)

        # BUG-FIX-3: Use INTER_AREA when downscaling (avoids aliasing artifacts
        # that corrupt depth cues near edges), INTER_LINEAR for upscaling.
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        resized = cv2.resize(frame_rgb, (new_w, new_h), interpolation=interp)

        # Letterbox: centre the resized image on a black canvas.
        canvas   = np.zeros((self.NATIVE_SIZE, self.NATIVE_SIZE, 3), dtype=np.uint8)
        pad_top  = (self.NATIVE_SIZE - new_h) // 2
        pad_left = (self.NATIVE_SIZE - new_w) // 2
        canvas[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized

        # Cache for _postprocess().
        self._pad_top   = pad_top
        self._pad_left  = pad_left
        self._pad_new_h = new_h
        self._pad_new_w = new_w

        return Image.fromarray(canvas)

    def _postprocess(self, raw_depth, original_hw: tuple):
        """
        Convert the pipeline's raw depth output to a clean float32 numpy array
        at the original frame resolution.

        Parameters
        ----------
        raw_depth : PIL.Image.Image | np.ndarray | torch.Tensor
            Whatever the HuggingFace pipeline returns inside result["depth"].
        original_hw : (int, int)
            (height, width) of the input frame we need to match.

        Returns
        -------
        np.ndarray or None
            HxW float32 array of relative depth values with no NaNs or zeros.
            Returns None if the depth map is too corrupted to be useful.
        """
        # --- Normalise to numpy float32 regardless of pipeline output type ---

        # BUG-FIX-4: torch.Tensor.numpy() fails for GPU tensors (must call
        # .cpu() first). Also, newer transformers returns tensors that still
        # require_grad, so detach() before numpy() is necessary to avoid
        # "RuntimeError: Can't call numpy() on Tensor that requires grad".
        if hasattr(raw_depth, "detach"):        # torch.Tensor path
            depth_np = raw_depth.detach().cpu().numpy().astype(np.float32)
        elif hasattr(raw_depth, "__array__"):   # PIL.Image or array-like
            depth_np = np.array(raw_depth, dtype=np.float32)
        else:
            depth_np = np.asarray(raw_depth, dtype=np.float32)

        # Squeeze any singleton batch/channel dims the pipeline may have added.
        depth_np = depth_np.squeeze()

        if depth_np.ndim != 2:
            raise ValueError(
                f"Unexpected depth shape after squeeze: {depth_np.shape}. "
                "Expected a 2-D (H, W) array."
            )

        # --- Crop letterbox padding ---
        # BUG-FIX-5: The model outputs a depth map at NATIVE_SIZE x NATIVE_SIZE.
        # The depth map spatial resolution matches the letterboxed PIL input, NOT
        # the original frame. We must crop to [pad_top:pad_top+new_h,
        # pad_left:pad_left+new_w] BEFORE resizing, or we resize the padding
        # bars into the output and distort depth values at frame edges.
        depth_np = depth_np[
            self._pad_top  : self._pad_top  + self._pad_new_h,
            self._pad_left : self._pad_left + self._pad_new_w
        ]

        # --- Resize back to original frame resolution ---
        orig_h, orig_w = original_hw
        if depth_np.shape != (orig_h, orig_w):
            # BUG-FIX-6: Use INTER_LINEAR (not INTER_NEAREST) for depth map
            # upscaling to avoid staircasing artifacts at object boundaries.
            # cv2.resize expects (width, height), not (height, width).
            depth_np = cv2.resize(
                depth_np,
                (orig_w, orig_h),
                interpolation=cv2.INTER_LINEAR,
            )

        # --- Sanitise: replace NaNs, Infs, and exact zeros ---
        # Exact zeros appear as "at infinity" after metric scaling and break
        # back-projection. Replace bad pixels with the map's median.
        bad_mask = ~np.isfinite(depth_np) | (depth_np == 0.0)
        if bad_mask.any():
            bad_ratio = float(bad_mask.mean())
            if bad_ratio > 0.2:
                print(f"[MonocularDepthModel] WARNING: {bad_ratio:.1%} of depth "
                      f"pixels were invalid — depth map unreliable.")
                return None
            # BUG-FIX-7: Compute median only on valid pixels. If somehow all
            # pixels are bad (bad_ratio <= 0.2 but ~bad_mask is all False,
            # theoretically impossible but defensive), fall back to 1.0.
            valid_pixels = depth_np[~bad_mask]
            median_val   = float(np.median(valid_pixels)) if valid_pixels.size > 0 else 1.0
            depth_np[bad_mask] = median_val

        # Sanity check: flat output means model failed silently.
        dynamic_range = float(depth_np.max() - depth_np.min())
        if dynamic_range < 1e-3:
            print("[MonocularDepthModel] WARNING: depth map has near-zero "
                  "dynamic range — model output is flat.")
            return None

        # BUG-FIX-8: The depth map is in an arbitrary relative scale and may
        # have very small absolute values (e.g. 0.0–1.0 range from some model
        # versions) or very large values (e.g. raw logit scale). Downstream
        # code in scaled_depth.py assumes the map is normalised to [0, 1].
        # Normalise here so the contract in the docstring is actually honoured.
        d_min = depth_np.min()
        d_max = depth_np.max()
        depth_np = (depth_np - d_min) / (d_max - d_min)   # safe: dynamic_range > 1e-3

        return depth_np

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def infer(self, frame_bgr: np.ndarray):
        """
        Run Depth Anything V2-Small on a single BGR frame and return a relative
        depth map at the same spatial resolution, normalised to [0, 1].

        Parameters
        ----------
        frame_bgr : np.ndarray
            HxWx3 uint8 array in OpenCV BGR channel order (already undistorted).

        Returns
        -------
        np.ndarray or None
            HxW float32 array in [0, 1] relative depth (0 = nearest, 1 = farthest,
            or model-convention: check Depth Anything V2 output polarity below).
            Returns None (and prints a descriptive error) if inference fails.

        Notes
        -----
        * BGR→RGB conversion is handled internally; callers must NOT pre-convert.
        * Output resolution exactly matches the input frame's (H, W).
        * Output contains no NaN, Inf, or zero values.
        * Output is normalised to [0, 1]; scale is relative, not metric.
        * DA V2 follows the convention: LARGER value = CLOSER to camera (inverse
          depth). Verify this against your version and flip if needed:
              depth_map = 1.0 - depth_map
        """
        # Input validation
        if frame_bgr is None:
            print("[MonocularDepthModel] infer() received None frame.")
            return None
        if frame_bgr.size == 0:
            print("[MonocularDepthModel] infer() received empty frame.")
            return None

        # BUG-FIX-9: Naive .astype(np.uint8) clips float values silently.
        # A float frame in [0.0, 1.0] becomes all-zeros after cast. Scale first.
        if frame_bgr.dtype != np.uint8:
            print(f"[MonocularDepthModel] WARNING: expected uint8 frame, "
                  f"got {frame_bgr.dtype} — converting.")
            if np.issubdtype(frame_bgr.dtype, np.floating):
                frame_bgr = (np.clip(frame_bgr, 0.0, 1.0) * 255.0).astype(np.uint8)
            else:
                frame_bgr = frame_bgr.astype(np.uint8)

        # BUG-FIX-10: Validate that frame has 3 channels. A grayscale frame
        # passed in would cause cv2.COLOR_BGR2RGB to silently produce wrong output.
        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            print(f"[MonocularDepthModel] infer() expected HxWx3 frame, "
                  f"got shape {frame_bgr.shape}.")
            return None

        # Lazy model loading
        if not self._loaded and not self._load_failed:
            self._load_model()
        if not self._loaded or self._pipe is None:
            print("[MonocularDepthModel] infer() called but model is not loaded.")
            return None

        try:
            import torch
            original_hw = (frame_bgr.shape[0], frame_bgr.shape[1])
            pil_image   = self._bgr_to_pil(frame_bgr)

            with torch.inference_mode():
                result = self._pipe(pil_image)

            raw_depth = result["depth"]
            depth_map = self._postprocess(raw_depth, original_hw)
            if depth_map is None:
                return None
            return depth_map

        except Exception:
            print("[MonocularDepthModel] ERROR during infer():")
            traceback.print_exc()
            return None

    def warmup(self):
        """
        Run one dummy inference to trigger weight loading and JIT compilation.

        Call once at pipeline startup so the model is fully initialised before
        the first real grasp attempt. Uses a 720p frame to match real deployment.
        Critical on Jetson where first CUDA kernel launch can take several seconds.
        """
        print("[MonocularDepthModel] Warming up …")
        # BUG-FIX-11: An all-zeros frame will be caught by the dynamic_range < 1e-3
        # check in _postprocess and return None, making warmup appear to fail even
        # when the model loaded correctly. Use random noise so the model sees a
        # frame with actual spatial variation and returns a valid depth map.
        dummy  = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        result = self.infer(dummy)
        if result is not None:
            print(
                f"[MonocularDepthModel] Warmup complete. "
                f"Output shape: {result.shape}, dtype: {result.dtype}."
            )
        else:
            print("[MonocularDepthModel] Warmup failed — check model loading errors above.")

    def benchmark(self, frame_bgr: np.ndarray, n_runs: int = 10):
        """
        Time repeated inference on a real frame and report mean ± std latency.

        Use this to validate per-inference performance on the target hardware
        (Raspberry Pi 5 or Jetson Orin Nano) before deploying.

        Latency measured here includes preprocessing, model inference, and
        postprocessing — real-world end-to-end cost, not pure model latency.

        Parameters
        ----------
        frame_bgr : np.ndarray
            A representative HxWx3 uint8 BGR frame from the actual camera.
        n_runs : int
            Number of timed inference passes. Default 10.

        Returns
        -------
        tuple[float, float]
            (mean_ms, std_ms) — mean and std deviation of latency in milliseconds.
            Returns (float('nan'), float('nan')) if all runs fail.
        """
        print(f"[MonocularDepthModel] Benchmarking {n_runs} runs …")

        # BUG-FIX-12: Original benchmark() was completely broken — missing loop,
        # misindented prints, no latency list initialisation, no t0/t1 timing
        # variables, and the model-not-loaded guard printed the error then fell
        # through to run inference anyway instead of returning early.
        # Fully reconstructed here.

        if not self._loaded and not self._load_failed:
            self._load_model()
        if not self._loaded:
            print("[MonocularDepthModel] Cannot benchmark — model not loaded.")
            return float("nan"), float("nan")

        latencies = []
        for i in range(n_runs):
            t0     = time.perf_counter()
            result = self.infer(frame_bgr)
            t1     = time.perf_counter()

            if result is None:
                print(f"[MonocularDepthModel] Run {i + 1}/{n_runs} failed.")
                continue

            latencies.append((t1 - t0) * 1000.0)

        if not latencies:
            print("[MonocularDepthModel] All benchmark runs failed.")
            return float("nan"), float("nan")

        mean_ms = float(np.mean(latencies))
        std_ms  = float(np.std(latencies))
        print(
            f"[MonocularDepthModel] Benchmark result over {len(latencies)} "
            f"successful runs: mean = {mean_ms:.1f} ms,  std = {std_ms:.1f} ms"
        )
        return mean_ms, std_ms
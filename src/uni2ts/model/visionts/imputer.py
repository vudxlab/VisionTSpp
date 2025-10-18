from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple

import numpy as np
import torch

from .model import VisionTS


@dataclass(frozen=True)
class MissingInterval:
    """
    Represents a contiguous missing span within a time series.

    Attributes
    ----------
    start : int
        Index (0-based) at which the missing span begins.
    length : int
        Number of consecutive points that are missing from ``start``.
    """

    start: int
    length: int

    @property
    def end(self) -> int:
        """Inclusive end index (exclusive upper bound)."""

        return self.start + self.length


class VisionTSMissingImputer:
    """
    Utility wrapper that turns the VisionTS forecasting backbone
    into a gap-filling (missing-value imputation) model.

    The class expects that the VisionTS instance has already been created
    and its checkpoint loaded. During imputation we repeatedly:
      1. gather a window of observed history in front of each missing span,
      2. forecast the next ``length`` points,
      3. write those predictions back into the series.

    Parameters
    ----------
    model : VisionTS
        Initialised VisionTS model. The wrapper will put it into ``eval`` mode.
    device : str, default "cpu"
        Device to move the model and inputs onto during imputation.
    max_context : int | None, default None
        Optional cap for the amount of historical context (in number of samples)
        to feed into the model. If ``None`` the entire available history before
        a gap is used.
    max_horizon : int, default 512
        Maximum number of future steps to predict in a single forward call.
        Longer gaps are automatically chunked into multiple smaller forecasts.
    norm_const : float, default 0.4
        Normalisation constant to pass to ``VisionTS.update_config``. This should
        match the value used during training.
    periodicity : int, default 1
        Base periodicity of the data (number of samples per cycle). This value
        is only used to build the internal masking strategy and should mirror
        the configuration used when fine-tuning the model.
    """

    def __init__(
        self,
        model: VisionTS,
        *,
        device: str = "cpu",
        max_context: int | None = None,
        max_horizon: int = 512,
        norm_const: float = 0.4,
        periodicity: int = 1,
    ) -> None:
        self.model = model.to(device)
        self.model.eval()
        self.device = torch.device(device)
        self.max_context = max_context
        self.max_horizon = max_horizon
        self.norm_const = norm_const
        self.periodicity = periodicity

    def _prepare_context(
        self,
        series: np.ndarray,
        start: int,
    ) -> np.ndarray | None:
        """
        Slice the most recent observed context in front of ``start``.

        Returns
        -------
        np.ndarray | None
            Array with shape ``(context_len, nvars)`` or ``None`` if not enough
            history is available to run the model.
        """

        if start <= 0:
            return None

        ctx_begin = 0 if self.max_context is None else max(0, start - self.max_context)
        context = series[ctx_begin:start]
        if context.size == 0:
            return None
        return context

    @staticmethod
    def _ensure_2d(arr: np.ndarray) -> np.ndarray:
        """
        Convert a 1D series into shape (time, 1) so VisionTS can consume it.
        """

        if arr.ndim == 1:
            return np.expand_dims(arr, -1)
        if arr.ndim != 2:
            raise ValueError(
                f"Expected 1D or 2D array for time series, received shape {arr.shape}"
            )
        return arr

    def _fill_single_gap(
        self,
        series: np.ndarray,
        interval: MissingInterval,
        *,
        fp64: bool = False,
    ) -> np.ndarray:
        """
        Forecast and write back predictions for a single missing span.
        """

        remaining = int(interval.length)
        if remaining <= 0:
            return series

        cursor = interval.start
        while remaining > 0:
            chunk = remaining if self.max_horizon is None else min(remaining, self.max_horizon)

            context = self._prepare_context(series, cursor)
            if context is None:
                break
            context = self._ensure_2d(context)
            context_len = context.shape[0]

            self.model.update_config(
                context_len=context_len,
                pred_len=chunk,
                periodicity=self.periodicity,
                norm_const=self.norm_const,
            )

            tensor = (
                torch.from_numpy(context.astype(np.float32))
                .unsqueeze(0)
                .to(self.device)
            )
            with torch.no_grad():
                forecast = self.model(tensor, fp64=fp64)

            pred = forecast.squeeze(0).cpu().numpy()
            if pred.ndim == 1:
                pred = np.expand_dims(pred, -1)

            end = cursor + chunk
            series[cursor:end] = pred[:chunk]

            remaining -= chunk
            cursor = end

        return series

    def impute(
        self,
        series: np.ndarray,
        missing: Sequence[MissingInterval] | Iterable[Tuple[int, int]],
        *,
        fp64: bool = False,
    ) -> np.ndarray:
        """
        Fill every interval provided in ``missing`` and return a new array.

        Parameters
        ----------
        series : np.ndarray
            Input time series with shape ``(time,)`` or ``(time, nvars)``.
        missing : Sequence[MissingInterval] | Iterable[Tuple[int, int]]
            Iterable of gaps to fill. Tuples are interpreted as ``(start, length)``.
        fp64 : bool, default False
            Forward the ``fp64`` flag to ``VisionTS.forward`` which can mitigate
            numerical overflow on certain datasets.
        """

        series_arr = np.array(series, copy=True)
        series_arr = self._ensure_2d(series_arr)

        intervals: list[MissingInterval] = []
        for interval in missing:
            if isinstance(interval, MissingInterval):
                intervals.append(interval)
            else:
                start, length = interval
                intervals.append(MissingInterval(int(start), int(length)))

        for interval in intervals:
            series_arr = self._fill_single_gap(series_arr, interval, fp64=fp64)

        if series_arr.shape[1] == 1:
            return series_arr[:, 0]
        return series_arr

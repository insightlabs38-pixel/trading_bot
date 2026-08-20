"""Exchange-calendar helpers for production session completeness and resampling."""

from __future__ import annotations

from datetime import date
from zoneinfo import ZoneInfo

from trading_bot.data.resampling import SessionSpec


class CalendarSessionError(ValueError):
    """Raised when a requested date is not a valid exchange trading session."""


class ExchangeCalendarSessionProvider:
    """Resolve trading dates and per-session hours from ``exchange_calendars``."""

    def __init__(
        self,
        calendar_name: str = "XNYS",
        *,
        base_interval_minutes: int = 1,
    ) -> None:
        if not calendar_name.strip():
            raise ValueError("calendar_name must not be blank")
        if base_interval_minutes <= 0:
            raise ValueError("base_interval_minutes must be positive")
        try:
            import exchange_calendars as xcals
        except ImportError as exc:  # pragma: no cover - exercised in core-only environments
            raise RuntimeError(
                "exchange-calendars is required for production exchange session support"
            ) from exc
        try:
            calendar = xcals.get_calendar(calendar_name)
        except (KeyError, ValueError) as exc:
            raise ValueError(f"unknown exchange calendar: {calendar_name!r}") from exc
        self.calendar_name = calendar_name
        self.base_interval_minutes = base_interval_minutes
        self._calendar = calendar
        self.timezone = str(calendar.tz)
        ZoneInfo(self.timezone)

    def session_dates(self, start: date, end: date) -> tuple[date, ...]:
        """Return actual exchange sessions in the inclusive date range."""
        if end < start:
            raise ValueError("session date range end cannot precede start")
        sessions = self._calendar.sessions_in_range(start, end)
        return tuple(value.date() for value in sessions)

    def is_session(self, session_date: date) -> bool:
        """Return whether ``session_date`` is an exchange trading session."""
        return bool(self._calendar.is_session(session_date))

    def session_spec(self, session_date: date) -> SessionSpec:
        """Return exact local open/close hours, including exchange early closes."""
        if not self.is_session(session_date):
            raise CalendarSessionError(
                f"{session_date.isoformat()} is not a {self.calendar_name} trading session"
            )
        open_utc = self._calendar.session_open(session_date).to_pydatetime()
        close_utc = self._calendar.session_close(session_date).to_pydatetime()
        timezone = ZoneInfo(self.timezone)
        local_open = open_utc.astimezone(timezone)
        local_close = close_utc.astimezone(timezone)
        if local_open.date() != session_date or local_close.date() != session_date:
            raise CalendarSessionError(
                "exchange calendar produced a session that crosses the configured local date"
            )
        return SessionSpec(
            timezone=self.timezone,
            open_time=local_open.time(),
            close_time=local_close.time(),
            base_interval_minutes=self.base_interval_minutes,
        )

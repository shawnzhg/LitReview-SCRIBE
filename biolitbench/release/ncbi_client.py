"""Shared NCBI client: one process-wide rate limit, tool and email parameters, backoff and the off-peak rule."""

from __future__ import annotations

import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
TOOL = "biolitbench"
MAX_IDS_PER_POST = 150
SERIES_LIMIT = 100
RATE_DEFAULT = 3.0
RATE_WITH_KEY = 10.0


class NcbiError(Exception):
    pass


class PeakHoursError(NcbiError):
    pass


class TokenBucket:
    def __init__(self, clock=time.monotonic, sleep=time.sleep):
        self.clock = clock
        self.sleep = sleep
        self.lock = threading.Lock()
        self.tokens = 1.0
        self.stamp = None

    def acquire(self, rate: float) -> float:
        with self.lock:
            now = self.clock()
            if self.stamp is None:
                self.stamp = now
            self.tokens = min(1.0, self.tokens + max(0.0, now - self.stamp) * rate)
            self.stamp = max(self.stamp, now)
            wait = 0.0
            if self.tokens < 1.0:
                wait = (1.0 - self.tokens) / rate
                self.sleep(wait)
                self.stamp += wait
                self.tokens = 1.0
            self.tokens -= 1.0
            return wait


LIMITER = TokenBucket()


def _nth_sunday(year: int, month: int, n: int) -> datetime:
    first = datetime(year, month, 1, tzinfo=timezone.utc)
    return first + timedelta(days=(6 - first.weekday()) % 7, weeks=n - 1)


def eastern_time(now: datetime) -> datetime:
    try:
        from zoneinfo import ZoneInfo
        return now.astimezone(ZoneInfo("America/New_York"))
    except Exception:
        u = now.astimezone(timezone.utc)
        dst = _nth_sunday(u.year, 3, 2) + timedelta(hours=7) <= u < _nth_sunday(u.year, 11, 1) + timedelta(hours=6)
        return u.astimezone(timezone(timedelta(hours=-4 if dst else -5)))


def off_peak(now: datetime | None = None) -> bool:
    e = eastern_time(now or datetime.now(timezone.utc))
    return e.weekday() >= 5 or e.hour >= 21 or e.hour < 5


def urllib_transport(method, url, data, headers, timeout):
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, dict(r.headers), r.read()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers or {}), b""


class NcbiClient:
    def __init__(self, email=None, api_key=None, allow_peak_hours=False, transport=None, limiter=None,
                 now=None, sleep=time.sleep, max_retries=6, backoff=1.0, timeout=300):
        self.email = email
        self.api_key = api_key
        self.allow_peak_hours = allow_peak_hours
        self.transport = transport or urllib_transport
        self.limiter = limiter or LIMITER
        self.now = now or (lambda: datetime.now(timezone.utc))
        self.sleep = sleep
        self.max_retries = max_retries
        self.backoff = backoff
        self.timeout = timeout
        self.rate = RATE_WITH_KEY if api_key else RATE_DEFAULT
        self.planned = 0
        self.n_eutils = 0

    def check_series(self) -> None:
        long_series = self.planned > SERIES_LIMIT or self.n_eutils > SERIES_LIMIT
        if long_series and not self.allow_peak_hours and not off_peak(self.now()):
            raise PeakHoursError(
                f"a series of more than {SERIES_LIMIT} E-utilities requests runs only on weekends or "
                "21:00-05:00 America/New_York; rerun then, or pass --allow-peak-hours")

    def plan(self, n_requests: int) -> None:
        self.planned += n_requests
        self.check_series()

    def eutils(self, endpoint: str, params: dict, check=None) -> bytes:
        ids = str(params.get("id", ""))
        if ids and len(ids.split(",")) > MAX_IDS_PER_POST:
            raise ValueError(f"at most {MAX_IDS_PER_POST} ids per request")
        body = dict(params)
        body["tool"] = TOOL
        if self.email:
            body["email"] = self.email
        if self.api_key:
            body["api_key"] = self.api_key
        data = urllib.parse.urlencode(body).encode("ascii")
        return self.request("POST", EUTILS + endpoint, data=data, check=check, eutils=True)[1]

    def get(self, url: str, accept=(200,), check=None):
        return self.request("GET", url, accept=accept, check=check)

    def request(self, method, url, data=None, accept=(200,), check=None, eutils=False):
        status = None
        for attempt in range(self.max_retries + 1):
            if eutils:
                self.n_eutils += 1
                self.check_series()
            self.limiter.acquire(self.rate)
            headers = {"User-Agent": TOOL}
            try:
                status, rheaders, body = self.transport(method, url, data, headers, self.timeout)
            except (urllib.error.URLError, OSError, TimeoutError):
                status, rheaders, body = None, {}, b""
            if status in accept and (status != 200 or check is None or check(body)):
                return status, body
            if status is not None and status != 429 and status < 500 and status not in accept:
                raise NcbiError(f"HTTP {status} from {url.split('?')[0]}")
            if attempt == self.max_retries:
                break
            delay = min(self.backoff * 2 ** attempt, 60.0)
            retry_after = str((rheaders or {}).get("Retry-After", "")).strip()
            if retry_after.isdigit():
                delay = max(delay, float(retry_after))
            self.sleep(delay)
        raise NcbiError(f"no valid response from {url.split('?')[0]} after {self.max_retries + 1} attempts "
                        f"(last status {status})")


def add_arguments(ap) -> None:
    ap.add_argument("--email")
    ap.add_argument("--api-key")
    ap.add_argument("--allow-peak-hours", action="store_true")


def from_args(args, **kw) -> NcbiClient:
    if not args.email:
        print("ncbi: --email is not set; NCBI asks distributed software to send a contact email",
              file=sys.stderr)
    return NcbiClient(email=args.email, api_key=args.api_key, allow_peak_hours=args.allow_peak_hours, **kw)

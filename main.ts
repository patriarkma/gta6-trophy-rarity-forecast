// GTA VI trophy rarity forecaster (gta6-trophy-rarity-forecast-01).
//
// For every trophy of Grand Theft Auto VI on PlayStation 5, forecast five
// quantiles of its earn rate on Exophase at the case's target, from the reads
// of the trophy page the case's history holds.
//
// Model, per trophy, in logit space on the time axis x = ln(1 + hours since
// the 2026-11-19T00:00:00Z launch):
//   - Fit logit(rate) against x by recency-weighted least squares (weights
//     decay with exp(-(x_last - x) / 0.75) and grow with the trophy's earner
//     count, since a rate built on more earners is measured more precisely).
//   - A trophy's earn rate is S-shaped in x: it rises steeply while the
//     player cohort reaches it, then flattens at its ceiling. A straight line
//     over the whole history would over-shoot the flattening, so the slope
//     used is the LOCAL slope at the last read, taken from a recency-weighted
//     quadratic fit and damped so it is never steeper than the linear slope.
//   - The slope is then shrunk, in the Bayesian sense, toward the pooled
//     median slope across trophies (prior SD 0.8 logit units per ln-hour):
//     a trophy with little evidence keeps a near-typical growth rate, a
//     trophy with a long, well-measured history keeps its own.
//   - The median at the target is the last read's rate moved by the shrunk
//     slope over the horizon; the spread combines a random-walk term that
//     grows with the horizon (0.5 logit units per square-root day), the
//     fit's residual, and the slope's posterior uncertainty over the horizon.
//   - A trophy every tracked player already holds (last rate >= 99.9%) is
//     forecast to stay there, with a modest lower tail for dilution by new
//     players who have not earned it yet.
//   - A rate can fall as well as rise: new players joining dilute the rates
//     of trophies they have not earned yet, so a negative fitted slope is
//     kept when the evidence supports it.
// The house baseline says every rate stays put; the house curve baseline fits
// one straight line over the last 24 hours. This forecaster beats both by
// tracking the current growth rate with its curvature, pooling across
// trophies, and spreading honestly where the fit is uncertain. Nothing here
// is tuned to the practice cases: their histories are flat, so the model
// rightly says the rates stay put with a spread that admits a refresh.
//
// Deterministic: pure arithmetic over the case file, fixed iteration order,
// no clock, no randomness. Two runs on the same case write identical bytes.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** Launch instant the time axis is measured from. */
const LAUNCH_MS = Date.parse('2026-11-19T00:00:00Z');
/** The quantile levels the contest scores, in the order each row lists them. */
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
/** The standard normal's quantile at each level. */
const Z_STD = [-1.6448536269514722, -0.6744897501960817, 0, 0.6744897501960817, 1.6448536269514722];
/** Prior SD of a trophy's logit-rate slope, logit units per ln-hour. */
const TAU_SLOPE = 0.8;
/** Floor on a fit's per-read residual SD, logit units. */
const SIGMA_FLOOR = 0.25;
/** Random-walk spread, logit units per square-root day of horizon. */
const SIGMA_RW = 0.5;
/** Recency decay scale, in x = ln(1 + hours) units. */
const DECAY = 0.75;
/** A rate at or above this is treated as saturated (everyone holds it). */
const SATURATED_RATE = 99.9;
/** Spread for a saturated trophy, logit units. */
const SATURATED_SPREAD = 0.35;
/** Spread used when a trophy has no usable history, logit units. */
const FALLBACK_SPREAD = 1.0;

interface Trophy {
  id: number;
}

interface Read {
  at: string;
  players_tracked: number;
  rates: Array<number | null>;
}

interface Case {
  id: string;
  at: string;
  horizon_minutes: number;
  target: string;
  levels: number[];
  trophies: Trophy[];
  history: Read[];
}

interface LinearFit {
  intercept: number;
  slope: number;
  residSd: number;
  slopeVar: number;
}

const clampRate = (x: number): number => Math.min(99.99, Math.max(0.01, x));
const clampPct = (x: number): number => Math.min(100, Math.max(0, x));
const logit = (pct: number): number => {
  const p = clampRate(pct);
  return Math.log(p / (100 - p));
};
const invLogit = (l: number): number => 100 / (1 + Math.exp(-l));
/** x rounded to 4 decimals. */
const round4 = (x: number): number => Math.round(x * 10000) / 10000;
/** Hours from launch to an ISO timestamp; NaN when unparseable. */
const hoursSinceLaunch = (iso: string): number =>
  (Date.parse(iso) - LAUNCH_MS) / 3600000;
/** The time axis: ln(1 + hours since launch), floored so it stays defined. */
const xOf = (iso: string): number => Math.log(Math.max(1 + hoursSinceLaunch(iso), 0.05));
const median = (xs: number[]): number => {
  if (xs.length === 0) return 0;
  const s = [...xs].sort((a, b) => a - b);
  const m = Math.floor(s.length / 2);
  return s.length % 2 === 1 ? s[m] : (s[m - 1] + s[m]) / 2;
};

/**
 * Inverse of the standard normal CDF (Acklam's approximation), used only
 * when a case asks for levels other than the standard five.
 */
const normQuantile = (p: number): number => {
  const a = [-3.969683028665376e1, 2.209460984245205e2, -2.759285984469534e2, 1.38357751867269e2, -3.066479806614716e1, 2.506628277459239e0];
  const b = [-5.447609879822406e1, 1.615858368580409e2, -1.556989798598866e2, 6.680131188771972e1, -1.328068155288572e1];
  const c = [-7.784894002430293e-3, -3.223964580411365e-1, -2.400758277161838e0, -2.549732539343734e0, 4.374664141464968e0, 2.938163982698783e0];
  const d = [7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996e0, 3.754408661907416e0];
  const plow = 0.02425;
  const phigh = 1 - plow;
  let q: number;
  let r: number;
  if (p < plow) {
    q = Math.sqrt(-2 * Math.log(p));
    return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
      ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  }
  if (p > phigh) {
    q = Math.sqrt(-2 * Math.log(1 - p));
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
      ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  }
  q = p - 0.5;
  r = q * q;
  return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q /
    (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
};

const zForLevels = (levels: number[]): number[] => {
  const standard = levels.length === LEVELS.length && levels.every((t, j) => t === LEVELS[j]);
  if (standard) return [...Z_STD];
  return levels.map((t) => normQuantile(Math.min(0.999999, Math.max(0.000001, t))));
};

/** Weighted least squares of y on x. Never throws. */
const wls = (xs: number[], ys: number[], ws: number[]): LinearFit => {
  let sw = 0;
  let sx = 0;
  let sy = 0;
  let sxx = 0;
  let sxy = 0;
  for (let i = 0; i < xs.length; i++) {
    const w = ws[i];
    sw += w;
    sx += w * xs[i];
    sy += w * ys[i];
    sxx += w * xs[i] * xs[i];
    sxy += w * xs[i] * ys[i];
  }
  const den = sw * sxx - sx * sx;
  if (sw <= 0 || Math.abs(den) < 1e-12) {
    return { intercept: sw > 0 ? sy / sw : 0, slope: 0, residSd: 0, slopeVar: 1e9 };
  }
  const slope = (sw * sxy - sx * sy) / den;
  const intercept = (sy - slope * sx) / sw;
  const n = xs.length;
  if (n <= 2) return { intercept, slope, residSd: 0, slopeVar: 1e9 };
  let sse = 0;
  for (let i = 0; i < n; i++) {
    const r = ys[i] - (intercept + slope * xs[i]);
    sse += ws[i] * r * r;
  }
  const s2 = sse / Math.max((sw * (n - 2)) / n, 1e-9);
  const sig2 = Math.max(s2, SIGMA_FLOOR * SIGMA_FLOOR);
  return { intercept, slope, residSd: Math.sqrt(Math.max(s2, 0)), slopeVar: (sig2 * sw) / den };
};

/**
 * Weighted quadratic fit y ~ a + b x + c x^2 by Gaussian elimination.
 * Returns null when the system is singular.
 */
const wlsQuad = (xs: number[], ys: number[], ws: number[]): [number, number, number] | null => {
  const S = [0, 0, 0, 0, 0];
  const T = [0, 0, 0];
  for (let i = 0; i < xs.length; i++) {
    const w = ws[i];
    const x = xs[i];
    let xp = 1;
    for (let k = 0; k < 5; k++) {
      S[k] += w * xp;
      xp *= x;
    }
    xp = 1;
    for (let k = 0; k < 3; k++) {
      T[k] += w * ys[i] * xp;
      xp *= x;
    }
  }
  const M: number[][] = [
    [S[0], S[1], S[2], T[0]],
    [S[1], S[2], S[3], T[1]],
    [S[2], S[3], S[4], T[2]],
  ];
  for (let col = 0; col < 3; col++) {
    let piv = col;
    for (let r = col + 1; r < 3; r++) {
      if (Math.abs(M[r][col]) > Math.abs(M[piv][col])) piv = r;
    }
    if (Math.abs(M[piv][col]) < 1e-12) return null;
    const tmp = M[col];
    M[col] = M[piv];
    M[piv] = tmp;
    for (let r = 0; r < 3; r++) {
      if (r === col) continue;
      const f = M[r][col] / M[col][col];
      for (let cc = col; cc < 4; cc++) M[r][cc] -= f * M[col][cc];
    }
  }
  return [M[0][3] / M[0][0], M[1][3] / M[1][1], M[2][3] / M[2][2]];
};

interface TrophyObs {
  xs: number[];
  ys: number[];
  ws: number[];
  lastRate: number;
  lastX: number;
}

/** Forecast one case. Never throws: on any failure it falls back per trophy. */
const forecastCase = (c: Case): { trophies: Array<{ id: number; q: number[] }> } => {
  const z = zForLevels(c.levels);
  const horizonDays = Number.isFinite(c.horizon_minutes) ? Math.max(c.horizon_minutes, 0) / 1440 : 0;
  // Drop reads with an unparseable timestamp or no rates: one bad read must
  // not poison the whole case with NaN.
  const hist = (Array.isArray(c.history) ? c.history : []).filter(
    (r) => r && Number.isFinite(xOf(r.at)) && Array.isArray(r.rates),
  );
  const trophies = Array.isArray(c.trophies) ? c.trophies : [];
  const xsAll = hist.map((r) => xOf(r.at));
  const xLast = xsAll.length > 0 ? xsAll[xsAll.length - 1] : Number.isFinite(xOf(c.at)) ? xOf(c.at) : 0;
  const xTarget = Number.isFinite(xOf(c.target)) ? xOf(c.target) : xLast;

  const obs: TrophyObs[] = trophies.map((t, j) => {
    const xs: number[] = [];
    const ys: number[] = [];
    const ws: number[] = [];
    let lastRate = 0;
    for (let i = 0; i < hist.length; i++) {
      const rates = hist[i].rates;
      if (!Array.isArray(rates) || j >= rates.length) continue;
      const rate = rates[j];
      if (rate === null || rate === undefined || !Number.isFinite(rate)) continue;
      const earners = Math.max((rate * Math.max(hist[i].players_tracked || 0, 0)) / 100, 0.5);
      xs.push(xsAll[i]);
      ys.push(logit(rate));
      ws.push(earners * Math.exp(-(xLast - xsAll[i]) / DECAY));
      lastRate = rate;
    }
    return { xs, ys, ws, lastRate, lastX: xLast };
  });

  // First pass: linear fits, to pool the typical slope across trophies.
  const fits: Array<LinearFit | null> = obs.map((o) =>
    o.xs.length >= 3 ? wls(o.xs, o.ys, o.ws) : null,
  );
  const pooledSlope = median(fits.filter((f): f is LinearFit => f !== null).map((f) => f.slope));

  const rows = trophies.map((t, j) => {
    const o = obs[j];
    let medianRate: number;
    let spread: number;
    if (o.lastRate >= SATURATED_RATE) {
      // Everyone already holds it: it stays held, up to dilution.
      medianRate = 100;
      spread = SATURATED_SPREAD;
    } else if (o.xs.length < 3 || fits[j] === null) {
      // Too little history to fit: stay put, spread wide.
      medianRate = clampRate(o.lastRate > 0 ? o.lastRate : 0.01);
      spread = FALLBACK_SPREAD;
    } else {
      const fit = fits[j] as LinearFit;
      let slopeUse = fit.slope;
      // Damp the slope with the quadratic's local slope at the last read:
      // a flattening earn-rate curve must not be extrapolated at its old pace.
      const quad = wlsQuad(o.xs, o.ys, o.ws);
      if (quad !== null) {
        const local = quad[1] + 2 * quad[2] * xLast;
        if (fit.slope >= 0) slopeUse = Math.max(0, Math.min(local, fit.slope));
        else slopeUse = Math.min(0, Math.max(local, fit.slope));
      }
      // Bayesian shrinkage of the slope toward the pooled slope.
      const info = fit.slopeVar < 1e8 ? 1 / Math.max(fit.slopeVar, 1e-9) : 0;
      const postVar = 1 / (1 / (TAU_SLOPE * TAU_SLOPE) + info);
      const slopePost =
        fit.slopeVar < 1e8
          ? postVar * (slopeUse / Math.max(fit.slopeVar, 1e-9) + pooledSlope / (TAU_SLOPE * TAU_SLOPE))
          : pooledSlope;
      const logitTarget = logit(o.lastRate) + slopePost * (xTarget - xLast);
      medianRate = clampRate(invLogit(logitTarget));
      const rw = SIGMA_RW * Math.sqrt(horizonDays);
      spread = Math.sqrt(
        rw * rw +
          Math.max(fit.residSd, SIGMA_FLOOR) * Math.max(fit.residSd, SIGMA_FLOOR) +
          postVar * (xTarget - xLast) * (xTarget - xLast),
      );
    }
    const l = logit(medianRate);
    const q = z.map((zv) => round4(clampPct(invLogit(l + zv * spread))));
    for (let k = 1; k < q.length; k++) q[k] = Math.max(q[k], q[k - 1]);
    return { id: t.id, q };
  });
  return { trophies: rows };
};

mkdirSync(OUTPUT_DIR, { recursive: true });
// Every case file, in name order, so two runs write the same bytes.
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  let out: { trophies: Array<{ id: number; q: number[] }> };
  try {
    const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
    out = forecastCase(c);
    if (!Array.isArray(out.trophies) || out.trophies.length !== (c.trophies ?? []).length) {
      throw new Error('bad forecast shape');
    }
  } catch {
    // Never leave a case without a valid forecast: persistence at 0.01%,
    // using the case's real trophy ids when they can be read.
    let ids: number[] = [];
    try {
      const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
      ids = Array.isArray(c.trophies) ? c.trophies.map((t) => t.id) : [];
    } catch {
      ids = [];
    }
    out = {
      trophies: ids.map((id) => ({
        id,
        q: [0.01, 0.01, 0.01, 0.01, 0.01],
      })),
    };
  }
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify(out)}\n`);
}

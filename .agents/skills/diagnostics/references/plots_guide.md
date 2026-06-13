# Diagnostic plots guide

What each plot in the diagnostics report should look like when the model is
healthy, and what the common pathologies look like.

## Residuals vs Fitted

**Healthy:** a structureless cloud centred on zero, with the LOWESS smoother
lying flat along the zero line. Spread roughly constant left to right.

**Funnel shape** (spread grows or shrinks with fitted values) →
heteroscedasticity. Cross-check with the Breusch-Pagan row in the assumption
table; remediation is robust SEs or a variance-stabilising transform.

**Curve in the LOWESS line** (U or inverted-U) → missed nonlinearity. The
model's mean function is wrong, not just its variance. Cross-check the Ramsey
RESET row; use partial residual plots to find which feature bends.

**Isolated extreme points** → candidates for the influence table; don't judge
them here, judge them on the leverage plot.

## Normal QQ plot

Sample residual quantiles vs theoretical normal quantiles. The dashed line
passes through the quartiles.

**Healthy:** points hug the line, small wiggles at the extreme ends are
normal (pun intended).

**Heavy tails** (points peel *above* the line at the right end and *below* at
the left) → outlier-prone errors; p-values are anti-conservative in small
samples. Consider robust regression or a transform.

**Light tails** (the opposite S-shape) → rarely harmful.

**Skew** (one tail deviates, the other doesn't) → usually fixed by a log or
Box-Cox transform of the target.

Mild non-normality with n in the hundreds is mostly cosmetic — the CLT
protects coefficient inference. It matters for *prediction intervals*.

## Scale-location

√|standardized residuals| vs fitted values — the variance-trend magnifier.

**Healthy:** flat LOWESS line, even vertical scatter.

**Upward-sloping LOWESS** → variance increases with the mean: classic
heteroscedasticity, often fixed with log(target). A downward slope is the
mirror image. This plot shows variance trends more clearly than residuals
vs fitted because the absolute value folds the residuals into one direction.

## Residuals vs Leverage

Studentized residuals vs leverage, marker size proportional to Cook's D.

**Healthy:** all points in a band |t| < 3, leverage below 2(k+1)/n, all
markers small.

**Large marker in the right half** → an observation that is *both* unusual in
X (high leverage) and poorly fitted (large residual): it is single-handedly
pulling the coefficients. This is the signature of the planted outlier in
`influential.csv`.

**High leverage, small residual** → an unusual-X point the line happens to
pass through; it inflates confidence in the slope without distorting it.
Verify it's a legitimate measurement.

**Large residual, low leverage** → an outlier in y only; it inflates the
residual variance but barely moves the fit.

Cook's D folds both axes into one number: D > 4/n is conventionally worth a
look; D exceeding that threshold by 5x or more is flagged HIGH by the triage
layer.

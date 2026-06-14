Output demo/regkit.gif

Set Shell "zsh"
Set FontSize 16
Set Width 1200
Set Height 680
Set Theme { "name": "regkit", "background": "#FFFFFF", "foreground": "#1F2937", "cursor": "#0F766E", "selectionBackground": "#5EEAD4", "black": "#1F2937", "red": "#DC2626", "green": "#059669", "yellow": "#D97706", "blue": "#0F766E", "magenta": "#7C3AED", "cyan": "#0D9488", "white": "#F9FAFB", "brightBlack": "#6B7280", "brightRed": "#EF4444", "brightGreen": "#10B981", "brightYellow": "#F59E0B", "brightBlue": "#14B8A6", "brightMagenta": "#8B5CF6", "brightCyan": "#5EEAD4", "brightWhite": "#FFFFFF" }
Set Padding 28
Set Framerate 24
Set TypingSpeed 50ms
Set WindowBar "Colorful"
Set BorderRadius 10

# ── silent setup ──────────────────────────────────────────────────────────────
Hide
Type "export PROMPT='%F{#0F766E}WindcleaverDev%f %F{#059669}❯%f '" Enter
Sleep 600ms
Type "uv run python tests/make_datasets.py > /dev/null 2>&1" Enter
Sleep 14s
Type "mkdir -p /tmp/rd" Enter
Sleep 1s
Type "clear" Enter
Sleep 400ms
Show

Sleep 800ms

# ── lasso: feature selection on diabetes ──────────────────────────────────────
Type "uv run python .agents/skills/regularized-regression/scripts/fit.py \"
Enter
Type "  --data tests/data/diabetes.csv --target progression \"
Enter
Type "  --features age,sex,bmi,bp,s1,s2,s3,s4,s5,s6 \"
Enter
Type "  --method lasso --alpha-rule 1se --output /tmp/rd/lasso"
Enter
Sleep 16s

Sleep 1400ms

# ── model comparison: nested LR test — bill vs bill + size ────────────────────
Type "uv run python .agents/skills/linear-regression/scripts/fit.py \"
Enter
Type "  --data tests/data/tips.csv --target tip --features total_bill \"
Enter
Type "  --output /tmp/rd/tips_bill"
Enter
Sleep 5s

Type "uv run python .agents/skills/linear-regression/scripts/fit.py \"
Enter
Type "  --data tests/data/tips.csv --target tip --features total_bill,size \"
Enter
Type "  --output /tmp/rd/tips_size"
Enter
Sleep 5s

Type "uv run python .agents/skills/model-comparison/scripts/compare.py \"
Enter
Type "  --reports /tmp/rd/tips_bill/report.json /tmp/rd/tips_size/report.json \"
Enter
Type "  --names 'OLS (bill)' 'OLS (bill+size)' --output /tmp/rd/tips_cmp"
Enter
Sleep 7s

Sleep 4s

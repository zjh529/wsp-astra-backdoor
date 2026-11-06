# Pi Weight-Space Programming (π · WSP) Terminal Demo
# Experiment 1: "Which game was this neural network trained on?"
# Recorded with Charm VHS (https://github.com/charmbracelet/vhs)

Output "docs/video/demo.mp4"
Output "docs/video/demo.gif"

Set Shell "zsh"
Set FontSize 14
Set Width 1320
Set Height 1040
Set WindowBar "Colorful"
Set BorderRadius 10
Set Theme "Catppuccin Mocha"
Set Padding 18
Set TypingSpeed 55ms

# 1. Start Pi in hidden mode so recording begins with Pi open & banner active
Hide
Type "export PATH=/Users/fcpenha/.local/bin:/opt/homebrew/bin:$PATH"
Enter
Type "cd /Users/fcpenha/git/pi-weight-space-programming"
Enter
Type "clear"
Enter
Type "pi -e extensions/index.ts --wsp-exp 1 --wsp-effort low"
Enter
Sleep 3500ms
Show

# 2. Pause on the banner so the viewer can take in the interface
Sleep 2500ms

# 3. Natural Language Prompt referencing the full 615-line weight prompt file from Experiment 1
Type "Which game was this neural network trained on? @evals/.cache/paper_evals/exp1/breakout_prompt.txt"
Sleep 800ms
# First Enter accepts the @file autocomplete selection; second Enter submits the prompt
Enter
Sleep 600ms
Enter

# 4. Real live run: Pi attaches the 17,253 weights and streams the mechanistic deduction
Sleep 25s

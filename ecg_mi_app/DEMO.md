# Demo checklist and script (16th)

## Day before
- [ ] `selftest.py` ends with **RESULT: PASS**
- [ ] `prepare_demo.py` has run; **Example ECGs** appears as the first input option
- [ ] Start the app with the desktop icon and click through **every** example once. Results are cached, so
      switching between them during the demo is instant.
- [ ] Note what the 🧩 hard case and the CSV upload show, so you know the result before going on stage
- [ ] Take screenshots of 3–4 screens as a backup (put them in the slides)
- [ ] Laptop: power adapter, automatic suspend off, notifications off, browser zoom ~110 % for the projector

## On the day
- [ ] Start the app **10 minutes before**. The first analysis loads the networks and takes longest.
- [ ] Open the first example once, so everything is warm.
- [ ] Wi-Fi is not needed.

## 5-minute script

1. **The task (30 s).** "Twelve-lead ECG in; out comes normal, inferior MI or anteroseptal MI, and *where*
   the evidence is." Show the **How it works** tab: four steps.
2. **A normal ECG (45 s).** Pick a ⭐ NORM example. Point at the result card, the green ✓ against the
   cardiologist label, and the three network votes.
3. **An inferior MI (1.5 min).** Pick a ⭐ IMI example.
   - *Median beats* tab: "Each lead's average heartbeat, drawn with its sign. This is the key idea of the
     model: ST elevation and Q waves stay visible."
   - *Lead importance* tab: check this the day before. If II, III or aVF are on top, say: "Hiding these
     leads changes the answer most. That is exactly where a cardiologist looks for an inferior MI, and the
     model was never told this." If not, pick the ⭐ IMI example where they are.
4. **An anteroseptal MI (45 s).** Pick a ⭐ ASMI example: the important leads move to V1–V4.
5. **A real-world file (45 s).** *Upload a recording* → the `ecg_machine_export_…csv`
   (500 Hz, microvolts, shuffled lead order). The notes show it was resampled and converted automatically.
6. **Honest limitation (30 s).** Pick the 🧩 hard case: an inferior MI called normal. "Inferior MI is the
   hardest class: 60 % recall on validation. Many are old infarcts with only small Q waves."
7. **Close (15 s).** "Runs offline on a laptop CPU in a few seconds per ECG (the result card shows the exact time). It is a research prototype, not a
   medical device."

## If something goes wrong
- **Browser shows an error page:** press **R** (rerun), or close the terminal and start again.
- **"Could not load the trained model":** the *Project folder* box in the sidebar must point to the folder
  that contains `runs/`.
- **Nothing works:** switch to the backup screenshots in the slides.

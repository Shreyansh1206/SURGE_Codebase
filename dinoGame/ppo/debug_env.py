"""
Step-by-step diagnostic.  Run:  python debug_env.py
"""
import sys, time, os
import numpy as np

_HERE     = os.path.dirname(os.path.abspath(__file__))
GAME_URL  = "file:///" + os.path.join(_HERE, "t-rex-runner", "index.html").replace("\\", "/")

print("=" * 60)
print("STAGE 1: imports")
print("=" * 60)
try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service
    print("  [OK] selenium")
except ImportError as e:
    print(f"  [FAIL] {e}  →  pip install selenium"); sys.exit(1)
try:
    from webdriver_manager.chrome import ChromeDriverManager
    HAS_WDM = True; print("  [OK] webdriver_manager")
except ImportError:
    HAS_WDM = False; print("  [WARN] webdriver_manager missing")

print()
print("=" * 60)
print("STAGE 2: launch Chrome")
print("=" * 60)
try:
    opts = Options()
    opts.add_argument("--mute-audio")
    opts.add_argument("--window-size=900,400")
    svc = Service(ChromeDriverManager().install()) if HAS_WDM else Service()
    driver = webdriver.Chrome(service=svc, options=opts)
    print("  [OK] Chrome launched")
except Exception as e:
    print(f"  [FAIL] {e}"); sys.exit(1)

print()
print("=" * 60)
print(f"STAGE 3: navigate to {GAME_URL}")
print("=" * 60)
try:
    driver.get(GAME_URL)
    print("  [OK] navigation succeeded")
except Exception as e:
    print(f"  [WARN] {type(e).__name__}: {e}")
time.sleep(1.5)
print(f"  title : {driver.title!r}")
print(f"  url   : {driver.current_url}")

print()
print("=" * 60)
print("STAGE 4: Runner constructor on page")
print("=" * 60)
info = driver.execute_script("""
return {
    runnerType : typeof Runner,
    instance_  : (typeof Runner !== 'undefined' && Runner.instance_ !== null)
                 ? 'non-null' : 'null',
    keycodes   : (typeof Runner !== 'undefined')
                 ? JSON.stringify(Runner.keycodes) : 'N/A',
};
""")
print(f"  typeof Runner    : {info['runnerType']}")
print(f"  Runner.instance_ : {info['instance_']}  (before any keypress)")
print(f"  Runner.keycodes  : {info['keycodes']}")
if info['runnerType'] == 'undefined':
    print("  [FAIL] Runner not defined — game JS didn't load")
    driver.quit(); sys.exit(1)

print()
print("=" * 60)
print("STAGE 5: probe what keyCode the game actually sees")
print("=" * 60)
received = driver.execute_script("""
let cap = null;
document.addEventListener('keydown', function _p(e) {
    cap = {keyCode:e.keyCode, key:e.key, code:e.code, which:e.which};
    document.removeEventListener('keydown', _p, true);
}, true);
document.dispatchEvent(new KeyboardEvent('keydown',
    {key:' ', code:'Space', keyCode:32, which:32, bubbles:true}));
return cap;
""")
print(f"  received: {received}")
ok_kc = received and received['keyCode'] == 32
print(f"  keyCode==32 : {'YES ✓' if ok_kc else 'NO ✗ — will fall back to direct call'}")

print()
print("=" * 60)
print("STAGE 6: kick game to start Runner.instance_")
print("=" * 60)
# Try key dispatch first; fall back to direct method call.
driver.execute_script("""
document.dispatchEvent(new KeyboardEvent('keydown',
    {key:' ', code:'Space', keyCode:32, which:32, bubbles:true}));
document.dispatchEvent(new KeyboardEvent('keyup',
    {key:' ', code:'Space', keyCode:32, which:32, bubbles:true}));
""")
time.sleep(0.6)
inst = driver.execute_script("return Runner.instance_ != null;")
print(f"  Runner.instance_ non-null after key dispatch: {inst}")

if not inst:
    print("  Trying direct onKeyDown call ...")
    driver.execute_script(
        "if (typeof Runner !== 'undefined' && Runner.instance_) {"
        "  Runner.instance_.onKeyDown({keyCode:32, preventDefault:()=>{}});"
        "}"
    )
    time.sleep(0.4)
    inst = driver.execute_script("return Runner.instance_ != null;")
    print(f"  After direct call: {inst}")

if not inst:
    print("  [FAIL] Runner.instance_ still null")
    driver.quit(); sys.exit(1)

print("  [OK] game running")

print()
print("=" * 60)
print("STAGE 7: read full state")
print("=" * 60)
s = driver.execute_script("""
const r = Runner.instance_;
if (!r) return null;
const t = r.tRex;
return {
    crashed : r.crashed,
    playing : r.playing,
    speed   : r.currentSpeed,
    dinoY   : t.yPos,
    jumping : t.jumping,
    score   : r.distanceMeter
              ? Math.floor(r.distanceRan * r.distanceMeter.config.COEFFICIENT)
              : -1,
};
""")
print(f"  state: {s}")
if not s:
    print("  [FAIL] null state"); driver.quit(); sys.exit(1)
print("  [OK]")

print()
print("=" * 60)
print("STAGE 8: 10 live steps  (watch the browser window)")
print("=" * 60)
SPACE_DN = "document.dispatchEvent(new KeyboardEvent('keydown',{key:' ',code:'Space',keyCode:32,which:32,bubbles:true}));"
SPACE_UP = "document.dispatchEvent(new KeyboardEvent('keyup',{key:' ',code:'Space',keyCode:32,which:32,bubbles:true}));"
DOWN_DN  = "document.dispatchEvent(new KeyboardEvent('keydown',{key:'ArrowDown',code:'ArrowDown',keyCode:40,which:40,bubbles:true}));"
DOWN_UP  = "document.dispatchEvent(new KeyboardEvent('keyup',{key:'ArrowDown',code:'ArrowDown',keyCode:40,which:40,bubbles:true}));"

for i in range(10):
    action = np.random.randint(0, 3)
    if   action == 1: driver.execute_script(SPACE_DN + SPACE_UP)
    elif action == 2: driver.execute_script(DOWN_DN)
    else:             driver.execute_script(DOWN_UP)
    time.sleep(4 / 60.0)
    s = driver.execute_script("""
    const r = Runner.instance_;
    if (!r) return null;
    return {
        crashed : r.crashed,
        score   : r.distanceMeter
                  ? Math.floor(r.distanceRan * r.distanceMeter.config.COEFFICIENT) : -1,
    };
    """)
    if not s: print(f"  step {i+1}: null state"); continue
    print(f"  step {i+1:2d}: action={action}  crashed={s['crashed']}  score={s['score']}")
    if s['crashed']:
        driver.execute_script("Runner.instance_.restart()"); time.sleep(0.4)

print()
print("=" * 60)
print("ALL STAGES PASSED — dino_env.py should work.")
print("Close the browser window now.")
print("=" * 60)
driver.quit()

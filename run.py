"""Start HandPilot.

    python run.py              open the app
    python run.py --stream     raw hand data, no mouse
    python run.py --doctor     check the connection to the phone
    python run.py --help       everything else
"""

from handpilot.__main__ import main

if __name__ == "__main__":
    raise SystemExit(main())

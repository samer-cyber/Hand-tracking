"""Start HandPilot.

    python run.py            open the window
    python run.py --doctor   check the connection path and exit
"""

import sys

if __name__ == "__main__":
    if "--doctor" in sys.argv:
        from tools.doctor import main as doctor
        raise SystemExit(doctor())
    from handpilot.ui import main
    main()

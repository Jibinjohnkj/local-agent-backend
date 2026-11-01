@echo off
rem Forwards %* to a C runtime program (python.exe) that prints its arguments as one JSON list.
rem Task 22.11: the Backend quotes batch arguments so that "%*" reaches the program unchanged.
rem The interpreter comes from ECHO_ARGS_PYTHON, which the test sets in the server's env.
"%ECHO_ARGS_PYTHON%" -c "import sys, json; print(json.dumps(sys.argv[1:]))" %*

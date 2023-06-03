@echo on
cd /d %~dp0 :: Change to directory of the bat file
call .\venv\Scripts\activate
pip freeze > requirements.txt
echo requirements.txt updated
pause

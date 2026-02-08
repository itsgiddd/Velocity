@echo off
echo Copying neural-forex-trading-app.zip to Desktop...
copy neural-forex-trading-app.zip "%USERPROFILE%\Desktop\"
if %ERRORLEVEL% EQU 0 (
    echo SUCCESS: ZIP file copied to Desktop!
    echo You can now find "neural-forex-trading-app.zip" on your Desktop
) else (
    echo ERROR: Could not copy to Desktop
    echo Trying alternative location...
    copy neural-forex-trading-app.zip "%USERPROFILE%\Downloads\"
    if %ERRORLEVEL% EQU 0 (
        echo SUCCESS: ZIP file copied to Downloads folder!
        echo You can now find "neural-forex-trading-app.zip" in your Downloads folder
    )
)
pause
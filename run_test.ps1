$process = Start-Process -FilePath "python" -ArgumentList "E:\AI\testing_cryp\app1.1\main.py" -RedirectStandardOutput "E:\AI\testing_cryp\app1.1\live_out.log" -RedirectStandardError "E:\AI\testing_cryp\app1.1\live_err.log" -PassThru -NoNewWindow
Start-Sleep -Seconds 75
Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
Get-Content "E:\AI\testing_cryp\app1.1\live_err.log"
Get-Content "E:\AI\testing_cryp\app1.1\live_out.log"

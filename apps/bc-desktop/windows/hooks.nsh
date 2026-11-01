; Installer hooks for the Windows app (bundle.windows.nsis.installerHooks).
;
; Check for updates starts this installer and then exits the app, so the app can still be
; shutting down when the installer looks for it. Tauri's own check then fails with "Failed
; to kill Borrowed Compute" and the update stops. Before that check, wait for this user's
; copy of the app to exit, and force it closed if it is still there after 10 seconds.

!macro BC_CLOSE_APP
  Push $R7
  Push $R8
  Push $R9
  StrCpy $R8 0
  ${Do}
    ; find exits 0 while the app is in this user's task list.
    nsExec::ExecToStack `"$SYSDIR\cmd.exe" /c "tasklist /NH /FI "IMAGENAME eq ${MAINBINARYNAME}.exe" /FI "USERNAME eq %USERDOMAIN%\%USERNAME%" | find /I "${MAINBINARYNAME}.exe""`
    Pop $R9
    Pop $R7
    ${If} $R9 <> 0
      ${Break}
    ${EndIf}
    ${If} $R8 = 20
      nsExec::ExecToStack `"$SYSDIR\cmd.exe" /c "taskkill /F /T /IM "${MAINBINARYNAME}.exe" /FI "USERNAME eq %USERDOMAIN%\%USERNAME%""`
      Pop $R9
      Pop $R7
    ${EndIf}
    Sleep 500
    IntOp $R8 $R8 + 1
  ${LoopUntil} $R8 >= 40
  Pop $R9
  Pop $R8
  Pop $R7
!macroend

; Something other than the app can still hold the program file open even after the app
; has exited, such as File Explorer reading its icon. Tauri's check asks Windows to close
; whatever holds the file and gives up with "Failed to kill" when it cannot, so the install
; or uninstall stops with nothing running. Windows allows renaming a file that is open or
; even running, and a renamed file has no holders for that check to find. The old file is
; deleted now if nothing holds it, otherwise at the next restart.
!macro BC_SET_EXE_ASIDE
  Delete "$INSTDIR\${MAINBINARYNAME}.exe.old"
  Rename "$INSTDIR\${MAINBINARYNAME}.exe" "$INSTDIR\${MAINBINARYNAME}.exe.old"
  Delete /REBOOTOK "$INSTDIR\${MAINBINARYNAME}.exe.old"
!macroend

!macro NSIS_HOOK_PREINSTALL
  !insertmacro BC_CLOSE_APP
  !insertmacro BC_SET_EXE_ASIDE
!macroend

!macro NSIS_HOOK_PREUNINSTALL
  !insertmacro BC_CLOSE_APP
  !insertmacro BC_SET_EXE_ASIDE
!macroend

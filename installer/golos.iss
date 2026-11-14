; Golos — установщик (Inno Setup 6)
#ifndef AppVersion
  #define AppVersion "0.1.0"
#endif

[Setup]
AppId={{8C1F5B2A-3E7D-4C2B-9F51-6A0D7E4B9C11}
AppName=Golos
AppVersion={#AppVersion}
AppPublisher=Golos
DefaultDirName={localappdata}\Programs\Golos
DefaultGroupName=Golos
PrivilegesRequired=lowest
OutputDir=..\dist
OutputBaseFilename=Golos-Setup-{#AppVersion}
SetupIconFile=..\golos\icon.ico
UninstallDisplayIcon={app}\golos\icon.ico
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
DisableProgramGroupPage=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible

[Languages]
Name: "ru"; MessagesFile: "compiler:Languages\Russian.isl"
Name: "en"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Files]
Source: "..\golos\*.py"; DestDir: "{app}\golos"; Flags: ignoreversion
Source: "..\golos\icon.ico"; DestDir: "{app}\golos"; Flags: ignoreversion
Source: "..\requirements.txt"; DestDir: "{app}"; Flags: ignoreversion
Source: "setup_env.ps1"; DestDir: "{app}"; Flags: ignoreversion
Source: "uv.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "python\*"; DestDir: "{app}\python"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\Golos"; Filename: "{app}\python\pythonw.exe"; Parameters: "-m golos"; WorkingDir: "{app}"; IconFilename: "{app}\golos\icon.ico"
Name: "{group}\Golos — обновить компоненты"; Filename: "powershell.exe"; Parameters: "-NoProfile -ExecutionPolicy Bypass -File ""{app}\setup_env.ps1"""; WorkingDir: "{app}"
Name: "{group}\{cm:UninstallProgram,Golos}"; Filename: "{uninstallexe}"
Name: "{userdesktop}\Golos"; Filename: "{app}\python\pythonw.exe"; Parameters: "-m golos"; WorkingDir: "{app}"; IconFilename: "{app}\golos\icon.ico"; Tasks: desktopicon

[Run]
Filename: "powershell.exe"; Parameters: "-NoProfile -ExecutionPolicy Bypass -File ""{app}\setup_env.ps1"""; WorkingDir: "{app}"; StatusMsg: "Скачиваю и устанавливаю компоненты (3–4 ГБ, несколько минут)…"; Flags: waituntilterminated
Filename: "{app}\python\pythonw.exe"; Parameters: "-m golos"; WorkingDir: "{app}"; Description: "{cm:LaunchProgram,Golos}"; Flags: nowait postinstall skipifsilent

[InstallDelete]
; остатки неудачной установки 0.2.0 (Python от uv с junction-ссылками)
Type: filesandordirs; Name: "{app}\.venv"
Type: filesandordirs; Name: "{app}\python\cpython-*"
Type: files; Name: "{app}\python\.lock"

[UninstallDelete]
Type: filesandordirs; Name: "{app}\.venv"
Type: filesandordirs; Name: "{app}\python"
Type: filesandordirs; Name: "{app}\golos\__pycache__"

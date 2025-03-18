## Install WSL:

### Install WSL with Default Distributions:

Install the Ubuntu distribution for WSL:

```
wsl --install -d Ubuntu
```

Install the base WSL if not already included (usually you won't have to do this):

```
wsl --install --no-distribution
```

You will have to restart your computer to enable these installations.

## Run WSL Within Your Terminal:

```
wsl -d Ubuntu
```

This might take a while at first. You may have to set a username and password as well.

Sometimes you have to update WSL before running and this will fix most WSL issues:

```
wsl --update
```

Ubuntu distributions should include the git binary.

## Install Docker in WSL:

```
sudo apt update &&
sudo apt upgrade -y &&
sudo apt install -y ca-certificates curl gnupg &&
sudo install -m 0755 -d /etc/apt/ &&
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo tee /etc/apt/keyrings/docker.asc > /dev/null &&
sudo chmod a+r /etc/apt/keyrings/docker.asc &&
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null &&
sudo apt update &&
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin &&
sudo usermod -aG docker $USER &&
newgrp docker
```

## Clone the Git Repo in WSL:
```
git clone https://github.com/Liamayyy/team5-capstone.git &&
cd team5-capstone
```

## (Optional) Uninstall WSL

If you wish, you may uninstall WSL after you are done with these three commands (with shell as admin), but this isn't necessary:

```
dism.exe /online /disable-feature /featurename:Microsoft-Windows-Subsystem-Linux /norestart ;
dism.exe /online /disable-feature /featurename:VirtualMachinePlatform /norestart ;
Remove-Item -Recurse -Force $env:LOCALAPPDATA\Packages\CanonicalGroupLimited*
```

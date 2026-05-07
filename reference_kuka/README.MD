# Project X

This repository contains all programs related to connecting the neuroArmPlus Hand Controllers to the following robot manipulators.

- KUKA KR6 via Simulink
- neuroArm

These connections are created by implementing the nap-sdk in a middleware program that can facilitate the transmission of motion commands from NAP to manipulator, and force feedback from manipulator to NAP.

## TODO 
- Silence and mute buttons broken on the middleware
- better neuroarm plus UI
- Ideally neuroarm plus ping
- video metrics
- proper audio disconnection handling

## Ping
For the ping functionality, we need to enable the cap:
```
sudo sysctl -w net.ipv4.ping_group_range="0 2147483647"
```

## Download necessary Linux packages

Download and install Rust with the necessary dependencies
```
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
sudo apt install build-essential
```

Download and install other necessary software packages
```
sudo apt-get update -y
sudo apt-get install -y cmake libudev-dev libasound2-dev libfontconfig1-dev
```

Configuration
```
echo 'net.ipv4.ping_group_range = 0 2147483647' | sudo tee -a /etc/sysctl.conf && sudo sysctl -p
```

### Configuring SpaceMouse
```
# Create a udev rule for your SpaceMouse
echo 'KERNEL=="hidraw*", ATTRS{idVendor}=="046d", ATTRS{idProduct}=="c626", MODE="0666"' | sudo tee /etc/udev/rules.d/99-spacemouse.rules

# Reload udev rules
sudo udevadm control --reload-rules
sudo udevadm trigger

# Unplug and replug your SpaceMouse
```

## Building

### KUKA KR6 via Simulink
For controlling the KUKA robot, Simulink is used to tie together the following functionality
- End Effector:
  - Forceps
  - Load Cell
- KUKA KR6:
  - Moving the arm with joint velocities
  - Joint positions and angles

Simulink will receive the 

### neuroArm
This program creates a way for an external computer to mimic the communications between the original neuroArm hand controllers, and the main system controller (CSD). It essentially fools the entire system into thinking that the external computer is indeed the controllers, thus ensuring that all internal system safeties are intact an the robot can be operated safely. 

## Kuka KR6 via Simulink

This program is used to control the Kuka Robot with the NAP Hand Controllers. It can run on the Beckhoff computer itself, or any computer on the network.

### Supported runtimes:
- Windows
- Linux

### Connection Requirements
Ensure you are connected to the neuroarm lab network (Ethernet is ideal)

Simulink TCP server
- Send: 10.20.30.172:17001
- Receive: 0.0.0.0:2022

NAP UDP Server
- Send: 10.20.30.130:1337
- Receive from other computer: 0.0.0.0:2012
- Receive from Beckhoff: 10.20.30.131:2012

### Running the program
```
cd kuka-middleware
cargo run --release
```

## neuroArm
Controlling the neuroArm with the NAP Hand Controllers requires the following computer setup:

### NA1 Controller
This will run on a seperate computer that will be hardwired to the neuroArm main system controller (CSD - 192.168.1.201)

**Validate connection**
On the neuroArm CSD computer (The one that has the HMI), open cmd and run `arp -d` to allow an external computer to ping it

Then try to ping the controller computer 

**Run the controller**
```
cd na1-controller
cargo run --release
```sudo apt-get install bluez-alsa

### NA1 Middleware
This program must be run in the lab, either directly on the Beckhoff, or on a seperate computer on the network

```
cd na1-middleware
```
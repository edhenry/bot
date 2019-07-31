# OS Tweaks

This is a list of tweaks I've made to the bot OS. These should be addressed and a new `golden` image should be generated. We'll likely want this pipelined, eventually.

1. Disable Unattended Auto Upgrades

Ubuntu, after 16.04 enabled automatic unattended upgrades by default. To disable this edit the configuration file found in `/etc/apt/apt.conf.d/20auto-upgrades` as such :

```
APT::Periodic::Unattended-Upgrade "1";
```
to
```
APT::Periodic::Unattended-Upgrade "0";
```
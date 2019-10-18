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

2. Install Telegraf for telemetry 

Add influx repos

```
wget -qO- https://repos.influxdata.com/influxdb.key | sudo apt-key add -
source /etc/lsb-release
echo "deb https://repos.influxdata.com/${DISTRIB_ID,,} ${DISTRIB_CODENAME} stable" | sudo tee /etc/apt/sources.list.d/influxdb.list
```

Install Telegraf

```
sudo apt-get update && sudo apt-get instal telegraf
```

The configuration file for Telegraf found at `/etc/telegraf/telegraf.conf` will need to be modified for whatever plugins you'll want to enable and/or configure.
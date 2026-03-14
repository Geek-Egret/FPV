Linux_for_Tegra/bootloader/generic/BCT/tegra234-mb2-bct-misc-p3767-0000.dts
pinmux.dtsi 和 padvoltage.dtsi 复制到 Linux_for_Tegra/bootloader/generic/BCT/
gpio.dtsi 复制到 Linux_for_Tegra/bootloader/
sudo tar xpf Tegra_Linux_Sample-Root-Filesystem_R[你的版本]_aarch64.tbz2 -C rootfs/

# 自定义载板系统移植
## Nvidia SDK
[SDK下载链接](https://developer.nvidia.cn/embedded/jetson-linux-r3644)
- 驱动程序 / 驱动包（BSP）
- 示例根文件系统（需解压到 Linux_for_Tegra/rootfs）
- Bootlin 工具链 gcc 11.3
## 禁用eeprom
- 由于自定义载板上没有eeprom，因此需要将CVB eeprom禁用
进入`dtb/tegra234-mb2-bct-misc-p3767-0000.dts`,将该文件复制到`Linux_for_Tegra/bootloader/generic/BCT/`下
## 命令行烧录
```
sudo ./tools/kernel_flash/l4t_initrd_flash.sh \
  --external-device mmcblk0p1 \
  -c tools/kernel_flash/flash_l4t_external.xml \
  -p "-c bootloader/generic/cfg/flash_t234_qspi.xml" \
  --showlogs \
  --network usb0 \
  p3509-a02-p3767-0000 internal
```
## JETSON系统设置
- 关闭网卡省电模式`sudo iw dev {网卡名字} set power_save off`
- 关闭网卡省电模式`sudo vim /etc/NetworkManager/conf.d/default-wifi-powersave-on.conf`将3改为2
- 重启服务`sudo systemctl restart NetworkManager`

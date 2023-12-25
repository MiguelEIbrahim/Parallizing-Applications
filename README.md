# Parallel Day to Day Programing
This Read me filel suggests that I am about to embard on parallelizing all day to day recources 
We'll start with: 
- Speeding up the read and write speeds from SSDs and Hard Drives respectfully.
    Phison's PS5026-E26 Max14um Gen5 SSD has reached sequential read speeds of more than 14,000MB/s - Fastest theoretical ssd
    Seagate unveiled its dual-actuator MACH. 2 hard drive technology back in 2021 with the 3.5in Exos 2x14 – the fastest HDD released at the time with a data transfer rate of 524MB/s

    Theoretical speedup: 26.717

Computer Configuration:
- Windows 11
- intel core i7-11800H @ 2.3GHz
- 24GB DDR4 RAM
- 1TB NVMe SSD SAMSUNG MZVL21T0HCLR-00B00 - 5100 MB/s  write speed
- Mediatek Wi-Fi 6
- integrated intel UHD Graphics
- Discrete GPU - Nvidia laptop series 3070 - 8GB VRAM

HPC Config - With Help from Ghent University HPC nodes:
- Linux Based Systems - donphan Shell tmux
- AMD EPYC - 1 node, 8 cores
- 26.6 GiB RAM
- ?? Storage - maybe Seagate HDDs
- 10Gib Ethernet
- A100 GPUs - not very sure

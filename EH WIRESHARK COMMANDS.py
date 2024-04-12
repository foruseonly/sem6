#!/usr/bin/env python
# coding: utf-8

# In[ ]:


TYBSC CS SEM- VI EH Wireshark COMMAND
1.	wireshark: This command launches the Wireshark GUI application.
 
2.	wireshark -r C:\Users\Admin\Documents\capture.pcapng: Opens a capture file for analysis in Wireshark.
 
3.	wireshark -D: Lists the available network interfaces that can be captured.
 
4.	wireshark -i wifi: Starts capturing packets on the specified network interface.
 
5.	wireshark -f "tcp port 80 or tcp port 443": Applies a capture filter to limit the types of packets captured.
  

6.	wireshark -Y tcp: Applies a display filter to limit the packets displayed in the GUI.
 
7.	wireshark –R tcp -r C:\Users\Admin\Documents\capture.pcapng: Reads a capture file and applies a read filter to display only packets matching the filter.
 
8.	wireshark -z wlan,stat: The command wireshark -z wsp,stat -r C:\Users\Admin\Documents\capture.pcapng is used to perform statistics calculations on captured packets related to the Wireless Session Protocol (WSP). WSP is a protocol used in wireless networks, particularly in the context of mobile communication systems such as GSM (Global System for Mobile Communications) and CDMA (Code Division Multiple Access).
 
9.	wireshark -c 10 -r C:\Users\Admin\Documents\capture.pcapng: Displays only the specified number of packets from the capture file.
 
10.	>wireshark -b duration:60 -b files:10: Sets a maximum capture file size and uses a ring buffer for continuous capture.
 
11.	wireshark -z wlanm,stat: This command performs Wi-Fi statistics analysis on captured packets and generates statistics related to Wi-Fi networks. It displays information such as Wi-Fi channel utilization, access point statistics, and signal strength.
 
12.	wireshark -h: Displays the command-line options and usage information for Wireshark.
 
13.	wireshark -X http -r C:\Users\Admin\Documents\capture.pcapng: Extracts and displays specific protocol details from the capture file.
 
14.	wireshark -t -n -r C:\Users\Admin\Documents\capture.pcapng: This command displays packet details without resolving addresses (IP, MAC, etc.) and timestamps them. It can be useful when you want to quickly analyze packet content without the overhead of address resolution.
 
15.	wireshark -z conv,tcp -r C:\Users\Admin\Documents\capture.pcapng: Performs TCP conversation analysis on the capture file.
 
16.	wireshark -z io,stat,1,tcp -r C:\Users\Admin\Documents\capture.pcapng: Generates input/output statistics for specified packets in the capture file.
 
17.	wireshark -z expert -r C:\Users\Admin\Documents\capture.pcapng: Displays the expert information, highlighting potential issues or anomalies in the captured packets.
 
18.	wireshark -z wlan,stat -r C:\Users\Admin\Documents\capture.pcapng: Generates HTTP statistics for packets captured in the file.
 
19.	wireshark -z sip,stat -r C:\Users\Admin\Documents\capture.pcapng: Generates Session Initiation Protocol (SIP) statistics from captured packets.
 
20.	wireshark –z ncp,stat -r C:\Users\Admin\Documents\capture.pcapng: The command wireshark -z ncp,srt -r C:\Users\Admin\Documents\capture.pcapng is used to analyze and generate statistics related to the service response times of NCP (NetWare Core Protocol) within the captured packets in the specified capture.pcapng file.


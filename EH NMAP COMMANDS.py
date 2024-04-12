#!/usr/bin/env python
# coding: utf-8

# In[ ]:


1.	Basic Scan: 
nmap
2.	Intense Scan: 
nmap-T4-A-v
3.	Service Version Detection: 
nmap-sV www.mu.ac.in
4.	Operating System Detection
nmap-O www.mu.ac.in
5.	TCP SYN Scan: 
nmap -sS www.mu.ac.in
6.	UDP Scan: 
nmap -sU www.mu.ac.in
7.	Aggressive Scan: 
nmap -T4-A-v  
8.	Fast Scan:
nmap _f www.mu.ac.in
9.	Ping Scan: 
nmap -sn www.mu.ac.in
10.	Port Range Scan:
nmap -p 1-100 www.mu.ac.in
11.	Script Scan:
nmap --script default www.mu.ac.in
12.	Scan multiple targets:
nmap www.mu.ac.in www.google.com
13.	 Scan for a specific port: 
nmap -p 80 www.mu.ac.in
14.	Scan for IPv6: 
nmap -6 www.mu.ac.in
15.	Scan using a specific interface:
nmap -e eth0 www.mu.ac.in
16.	Scan using a list of targets from a file: 
nmap -iL "location of the file"
#file should contain#
www.google.com
www.facebook.com
www.mu.ac.in
17.	Aggressive Timing Scan: 
nmap -T4 www.mu.ac.in
18.	Scan for common vulnerabilities: 
nmap --script vuln www.mu.ac.in
19.	Scan for SSL/TLS vulnerabilities: 
nmap --script ssl-enum-ciphers www.mu.ac.in
20.	Scan using decoy IPs: 
nmap -D 192.0.78.1,192.168.24.7,www.mu.ac.in
 


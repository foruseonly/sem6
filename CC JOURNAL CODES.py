#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#prac2:foss cloud steps#


# In[ ]:


prac 2-FOSS

create vm
192.168.10.3->login page
username login
password login

open winscp 
hostname(192.168.10.3)
username root
password admin
login

ubuntu iso file 
drag and paste to var-virtualization-iso choosable
()

vm->create template
profile-->create linux1 name and fill details and click create
click green arrow 
then click cd 


# In[ ]:


#prac3:saas#


# In[ ]:


prac 3-Saas
administration->system and domain
username administrator
password admin@123
login

users management of domain
add new user
insert username and other details
next
set password->create user
logout

application->owncloud
password in user management
-----------------------------------------------
Go to the OwnCloud IP address
Click on System and Domain Settings
Login using Administrator & admin@123 as username & password
Click on Create New Users & Add new Users
Then Logout 
Click on OwnCloud application
And Login
using the Created User Credentials

#rss feed#
<h3><b>Choose category: </b></h3>
<form method="post" id="myform">
<select name="rssurl" required>
<option value="">Select</option>
<option value="http://timesofindia.indiatimes.com/rssfeeds/-2128672765.cms">Science</option>
<option value="https://timesofindia.indiatimes.com/rssfeeds/65857041.cms">Astrology</option>
<option value="http://timesofindia.indiatimes.com/rssfeeds/66949542.cms">Tech</option>
</select>
<input type="submit" value="Load"/>
</form>
<?php
if(isset($_POST['rssurl']))
{
echo '<h1> Search Result for RSS url: '.$_POST['rssurl'].'</h1>';
$rssurl=$_POST['rssurl'];
$rss=new DOMDocument();
$rss->load($rssurl);
$feed=array();
foreach($rss->getElementsByTagName('item')as $node)
{
$item=array('title'=>$node->getElementsByTagName('title')->item(0)->nodeValue, 'desc'=>$node->getElementsByTagName('description')->item(0)->nodeValue,  'link'=>$node->getElementsByTagName('link')->item(0)->nodeValue, 'date'=>$node->getElementsByTagName('pubDate')->item(0)->nodeValue);
array_push($feed,$item);
}
$limit=5;
for($x=0;$x<$limit;$x++)
{
$title=str_replace('&','&amp;',$feed[$x]['title']);
$link=$feed[$x]['link'];
$description=$feed[$x]['desc'];
$date=date('I F d,Y',strtotime($feed[$x]['date']));
echo '<p> <strong><a href="'.$link.'" title="'.$title.'">'.$title.'</a></strong><br>';
echo '<p>'.$description.' </p>';
echo '<small><em>Posted on '.$date.'</em></small>';
}
}
?>


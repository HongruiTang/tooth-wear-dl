# tooth-wear-dl
This is a deep learning model for tooth wear evaluation in clinics.

## Running Project

### Cloud Azure

As we host our database on the Azure Virtual Machine, you need to connect to our server before being able to use the application. 

**Step 1:**  Go to <a href="https://azure.microsoft.com/en-gb">Azure website</a> and create a virtual machine. Detailed guideline can be found <a href="https://learn.microsoft.com/en-us/azure/virtual-machines/linux/quick-create-portal?tabs=ubuntu">here</a>. Make sure to take note of the username, password, and your vm ip address for the virtual machine.

**Step 2:** Use SSH to connect to the cloud server. 

``` 
ssh username@your-vm-ip-address
```
  
**Step 3:** Once you are connected to the remote virtual machine, you will need to clone our repository.

```
git clone https://github.com/HongruiTang/tooth-wear-dl.git
```
    
**Step 4:** Download related packages from the file requirements.txt:
```
cd tooth-wear-dl
pip install -r requirements.txt
```

**Step 5:** Navigate to the directory where the app.py file is located.
```
cd back-end/app/
```

**Step 6:** Run the back-end code by using the following command:
```
python app.py
```

If you want to run the app back-end code continuously on the server, you can use the following command:
```
nohup python app.py
```
  
Now the back-end preparation is done. You can now open Wear3D after downloading here: https://github.com/TilenLS/Wear3D

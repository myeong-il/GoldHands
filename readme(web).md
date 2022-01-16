### 0. WEB Structure

```
  📁 static
  	📁 css
  		📄 main_style.css   
      📄 result_style.css
  	📁 image
  		🖼 Loading-bar1.gif
      🖼 Loading-bar2.gif
  	📁 js
  		📄 canvas_js.js
  	📁 predict   // image predict result
  	📁 uploads   // sketch upload
      
      
  📁 templates
  	📄 canvas.html
  	📄 result.html
  
  
  📁 train_results     // 저장된 model weight
    📁 AE_trial-pr-face-8-20-23-30
        📄 100000.pth
		📁 GAN_trial-pr-face-8-20-23-30
				📄 19.pth
				
				
  📁 imgStyle         // 이미지 생성시 참조할 style
  	📁 0000
    📁 0001
       :
       :
    📁 1111
  
  
  📄 benchmark_test.py
  📄 config.py
  📄 datasets.py  
  📄 models.py  
  📄 utils.py  
  
  📄 server.py  
```

- python server.py 를 통해 웹 시작 가능


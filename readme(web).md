### 0. WEB Structure

```
  π static
  	π css
  		π main_style.css   
      π result_style.css
  	π image
  		πΌ Loading-bar1.gif
      πΌ Loading-bar2.gif
  	π js
  		π canvas_js.js
  	π predict   // image predict result
  	π uploads   // sketch upload
      
      
  π templates
  	π canvas.html
  	π result.html
  
  
  π train_results     // μ μ₯λ model weight
    π AE_trial-pr-face-8-20-23-30
        π 100000.pth
		π GAN_trial-pr-face-8-20-23-30
				π 19.pth
				
				
  π imgStyle         // μ΄λ―Έμ§ μμ±μ μ°Έμ‘°ν  style
  	π 0000
    π 0001
       :
       :
    π 1111
  
  
  π benchmark_test.py
  π config.py
  π datasets.py  
  π models.py  
  π utils.py  
  
  π server.py  
```

- python server.py λ₯Ό ν΅ν΄ μΉ μμ κ°λ₯


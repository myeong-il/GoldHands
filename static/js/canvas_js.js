const canvas = document.getElementById("draw_canvas");

canvas.width = 480;
canvas.height = 520;

let context = canvas.getContext("2d");
let start_background_color = "white"; 
context.fillStyle = start_background_color;
context.fillRect(0, 0, canvas.width, canvas.height);

let draw_color = "black";
let draw_width = "2";
let is_drawing = false;

let restore_array = [];
let index = -1;

var cnt = 0;
var time = 40;

/* 스케치 색 변경 적용 함수 */
function change_color(element) {
    draw_color = element.style.background;
}

/* Canvas 스케치 그리는 함수 */
const realInput = document.querySelector('#real-input');
realInput.addEventListener('change', handleImage, false);
function handleImage(e){
    var reader = new FileReader();
    reader.onload = function(event){
        var img = new Image();
        img.onload = function(){
            canvas.width = img.width;
            canvas.height = img.height;
            context.drawImage(img,0,0);
        }
        img.src = event.target.result;
    }
    reader.readAsDataURL(e.target.files[0]);     
}

canvas.addEventListener("touchstart", start, false);
canvas.addEventListener("touchmove", draw, false);
canvas.addEventListener("mousedown", start, false);
canvas.addEventListener("mousemove", draw, false);

canvas.addEventListener("touchend", stop, false);
canvas.addEventListener("mouseup", stop, false);
canvas.addEventListener("mouseout", stop, false);


function start(event) {
    is_drawing = true;
    context.beginPath();
    context.moveTo(event.clientX - canvas.offsetLeft, event.clientY - canvas.offsetTop);
    event.preventDefault();
}

function draw(event) {
    if (is_drawing) {
        context.lineTo(event.clientX - canvas.offsetLeft, event.clientY - canvas.offsetTop);
        context.strokeStyle = draw_color;
        context.lineWidth = draw_width;
        context.lineCap = "round";
        context.lineJoin = "round";
        context.stroke();
    }
}

function stop(event) {
    if (is_drawing) {
        context.stroke();
        context.closePath();
        is_drawing = false;
    }
    event.preventDefault();

    if (event.type != 'mouseout') {
        restore_array.push(context.getImageData(0, 0, canvas.width, canvas.height));
        index += 1;
    }
}
/* Canvas 스케치 그리는 함수 */

/* Canvas 초기화 */
function clear_canvas() {
    
    canvas.width = 480;
    canvas.height = 520;
    context.fillStyle = start_background_color;

    context.clearRect(0, 0, canvas.width, canvas.height);
    context.fillRect(0, 0, canvas.width, canvas.height);
    
    restore_array = [];
    index = -1;
}

/* 되돌리기 */
function undo_last() {
    if (index <= 0) {
        clear_canvas();
    } else {
        canvas.width = 480;
        canvas.height = 520;
        context.fillRect(0, 0, canvas.width, canvas.height);
        index -= 1;
        restore_array.pop();
        context.putImageData(restore_array[index], 0, 0);
    }
}

/* Process 버튼 클릭시 이미지와 스타일 선택 요소 value를 웹서버를 통해 모델로 전달 후 로딩하는 동안 타이머 실행 */
function toDataURL(){
    clearInterval(cnt);
    document.getElementById('myImage').src = canvas.toDataURL();
    var image = new Image();
    var url = document.getElementById('url');
    var sex = document.getElementById('sex');
    var glass = document.getElementById('glass');
    var hair = document.getElementById('hair');
    var skin = document.getElementById('skin');
    /* Canvas 이미지 src 전달 */
    image.src = document.getElementById('myImage').src;
    url.value = image.src;

    /* 스타일 요소 선택 Value 저장 */
    sex.value = document.getElementById("gender").value;;
    glass.value = document.getElementById("glasses").value;
    hair.value = document.getElementById("hair_color").value;
    skin.value = document.getElementById("skin_color").value;

    /* 로딩화면 띄우고 타이머 실행 */
    document.getElementById('loader').style.display = "block";
    cnt = setInterval("myTimer()", 1000);
}

/* 로딩중 소요시간 표시 함수 */
function myTimer() {

    document.getElementById('Timer').innerHTML = "예상 소요시간 : " + time + "초";
    time--;

    if (time == 0) {
        clearInterval(cnt);
        document.getElementById('Timer').innerHTML = "잠시만 기다려주세요.";
    }
}

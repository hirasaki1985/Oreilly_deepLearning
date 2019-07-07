// var apiUrl = "http://" + location.host + ":8000";
var apiUrl = "http://127.0.0.1:8000";

$(function() {
  // view
  function changeAnswer(answer) {
    $('#answer').html(answer);
  };

  function clearAllSpace() {
    clearCan();
    $('#answer').html("");
  };

  // triggers
  $('#submit').click(function() {

    var canvasData = document.getElementById("testImage");
    var imgObj = canvasData.toDataURL('image/png');

    //var canvasData = $('testImage').get(0).toDataURL();
    pngImage = imgObj.replace(/^.*,/, '');

    postData(apiUrl, {'file':pngImage}, {
      done: function(data) {
        console.log(typeof(data));
        console.dir(data.y);
        console.dir(data.max);
        changeAnswer(data.max);
      }
    });
  });
  $('#clear').click(function() {
    clearAllSpace();
  });

  // modules
  function postData(url, params, callback) {
    console.log("postData() exec");
    console.dir(params);
    $.ajax({
      url:url,
      type:"POST",
      crossDomain : true,
      data: params,
      dataType:'json',
      /*
      success:function(data) {
        console.log("postData() success");
        var result = JSON.parse(data);
        console.dir(result);
        callback.done(result);
      }
      */
    }).done(function( data ) {
      console.log("postData() done");
      console.debug(data);
      callback.done(data);
    }).fail(function( data ) {
      console.log("postData() fail");
      console.debug(data);
      callback.fail(data);
    });
    return ;
  }

  var can;
  var ct;
  var ox=0,oy=0,x=0,y=0;
  var mf=false;

  function mam_draw_init(){
    //初期設定
    can=document.getElementById("testImage");
    can.addEventListener("touchstart",onDown,false);
    can.addEventListener("touchmove",onMove,false);
    can.addEventListener("touchend",onUp,false);
    can.addEventListener("mousedown",onMouseDown,false);
    can.addEventListener("mousemove",onMouseMove,false);
    can.addEventListener("mouseup",onMouseUp,false);
    ct=can.getContext("2d");
    ct.strokeStyle="#000000";
    ct.lineWidth=5;
    ct.lineJoin="round";
    ct.lineCap="round";
    clearCan();
  }
  function onDown(event){
    mf=true;
    ox=event.touches[0].pageX-event.target.getBoundingClientRect().left;
    oy=event.touches[0].pageY-event.target.getBoundingClientRect().top;
    event.stopPropagation();
  }
  function onMove(event){
    if(mf){
      x=event.touches[0].pageX-event.target.getBoundingClientRect().left;
      y=event.touches[0].pageY-event.target.getBoundingClientRect().top;
      drawLine();
      ox=x;
      oy=y;
      event.preventDefault();
      event.stopPropagation();
    }
  }
  function onUp(event){
    mf=false;
    event.stopPropagation();
  }

  function onMouseDown(event){
    ox=event.clientX-event.target.getBoundingClientRect().left;
    oy=event.clientY-event.target.getBoundingClientRect().top ;
    mf=true;
  }
  function onMouseMove(event){
    if(mf){
      x=event.clientX-event.target.getBoundingClientRect().left;
      y=event.clientY-event.target.getBoundingClientRect().top ;
      drawLine();
      ox=x;
      oy=y;
    }
  }
  function onMouseUp(event){
    mf=false;
  }
  function drawLine(){
    ct.beginPath();
    ct.moveTo(ox,oy);
    ct.lineTo(x,y);
    ct.stroke();
  }
  function clearCan(){
    ct.fillStyle="rgb(255,255,255)";
    ct.fillRect(0,0,can.getBoundingClientRect().width,can.getBoundingClientRect().height);
  }
  mam_draw_init();
});
var odTaskInfo = {"sessname":"", "key":""};

function change_origin_url() {
    $('.oriimg').css({'display':'block'});
    var base = "/objectdetection/odres/odiptimg"
      , sessUrl = "?task=" + odTaskInfo['sessname'] + "&key=" + odTaskInfo['key'];
    $('#oriimgbody').attr('src',base + sessUrl);
}

function change_result_notify(message) {
    var base = "http://placehold.jp/24/cccccc/ffffff/300x300.png?text=";
    $('#resimgbody').attr('src',base + message);    
}

function change_result_complete() {
    var base = "/objectdetection/odres/odresimg"
      , sessUrl = "?task=" + odTaskInfo['sessname'] + "&key=" + odTaskInfo['key'];
    $('#resimgbody').attr('src',base + sessUrl);    
}

function wait_for_od_complete(retryCnt) {
    var base = "/objectdetection/odres"
      , sessUrl = "?task=" + odTaskInfo['sessname'] + "&key=" + odTaskInfo['key'];
    $('.resimg').css({'display':'block'});
      
    $.ajax({
		url: base + sessUrl,
		type: 'get',
		data: {},
		error: function (xhr, ajaxOptions, thrownError) {
			callback(xhr.status + " " + thrownError + ". Cannot connect to " + base + ".");
		},
		success: function (response) {
			//console.log(response);
            if(['failure','complete'].indexOf(response['state']) < 0) {
                change_result_notify("state: " + response['state']);
                setTimeout(function(){ 
                    wait_for_od_complete(retryCnt); 
                    console.log("Keep waiting for the calculation complete.");
                }, 2000);
            } else if (response['state'] == "complete") {
                change_result_complete();
            } else if (response['state'] == "failure") {
                // the memory not enough might cause exception
                // but in the final it would success
                if(retryCnt < 5) {
                    setTimeout(function(){ 
                        wait_for_od_complete(retryCnt + 1); 
                        console.log("Keep waiting for the calculation complete.");
                    }, 2000);
                } else {
                    console.log("Processing failed.");
                }
            }
		}
	});
}

$(function(){
  $('#odupload').on('click', function() {
    $.ajax({
      url: '/objectdetection',
      type: 'POST',

      // Form data
      data: new FormData($('#objectdetection')[0]),

      // Tell jQuery not to process data or worry about content-type
      // You *must* include these options!
      cache: false,
      contentType: false,
      processData: false,

      // Custom XMLHttpRequest
      xhr: function() {
        myXhr = $.ajaxSettings.xhr();
        if (myXhr.upload) {
          // For handling the progress of the upload
          myXhr.upload.addEventListener('progress', function(e) {
          if (e.lengthComputable) {
            $('progress').attr({value: e.loaded, max: e.total});
          }} , false);
          myXhr.upload.addEventListener('loadend', function(e) {
            console.log("Uploading image is complete.");
          });
        }
        myXhr.onreadystatechange = function() {
          if (myXhr.readyState == 4 && myXhr.status == 200) {
            //console.log(myXhr.response);
            if(myXhr.response.length > 0) {
                var response = JSON.parse(myXhr.response);
                if(response['status'] == "success") {
                    odTaskInfo = {"sessname":response['sessname'], "key":response['sesskey']};
                    // change origin image url
                    change_origin_url();
                    change_result_notify('Initialize object detection.');
                    // wait for object detection calculation complete
                    wait_for_od_complete(0);
                }
            }
          }
        }
        return myXhr;
       }
    });
  });
});
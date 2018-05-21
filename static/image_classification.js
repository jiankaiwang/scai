var icTaskInfo = {"sessname":"", "key":""};

function ic_change_origin_url() {
    $('.classifyorigin').css({'display':'block'});
    var base = "/imageclassification/imgclassres/iciptimg"
      , sessUrl = "?task=" + icTaskInfo['sessname'] + "&key=" + icTaskInfo['key'];
    $('#classifyoriginbody').attr('src',base + sessUrl);
}

function ic_change_result_notify(message) {
    $('.icres').html(message); 
}

function wait_for_ic_complete() {
    var base = "/imageclassification/imgclassres"
      , sessUrl = "?task=" + icTaskInfo['sessname'] + "&key=" + icTaskInfo['key'];
      
    $.ajax({
		url: base + sessUrl,
		type: 'get',
		data: {},
		error: function (xhr, ajaxOptions, thrownError) {
			callback(xhr.status + " " + thrownError + ". Cannot connect to " + base + ".");
		},
		success: function (response) {
			//console.log(response['state']);
            if(['failure','complete'].indexOf(response['state']) < 0) {
                ic_change_result_notify("state: " + response['state']);
                setTimeout(function(){ 
                    wait_for_ic_complete(); 
                    console.log("Keep waiting for the calculation complete.");
                }, 2000);
            } else if (response['state'] == "complete") {
                //ic_change_result_notify("finish");
                //console.log(JSON.parse(response['result']));
                var allres = JSON.parse(response['result']);
                var keyList = getDictionaryKeyList(allres);
                var showResMsg = '';
                for(var i = 0 ; i < keyList.length; i++) {
                    showResMsg += keyList[i] + ": " + allres[keyList[i]] + "<br>";
                }
                ic_change_result_notify(showResMsg);
            } else if (response['state'] == "failure") {
                ic_change_result_notify("failure to classify the image");
            }
		}
	});
}

$(function(){
  $('#icupload').on('click', function() {
    $.ajax({
      url: '/imageclassification',
      type: 'POST',

      // Form data
      data: new FormData($('#imageclassification')[0]),

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
                    icTaskInfo = {"sessname":response['sessname'], "key":response['sesskey']};
                    // change origin image url
                    ic_change_origin_url();
                    ic_change_result_notify('Initialize image classification.');
                    // wait for object detection calculation complete
                    wait_for_ic_complete();
                }
            }
          }
        }
        return myXhr;
       }
    });
  });
});
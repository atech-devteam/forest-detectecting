    Dropzone.autoDiscover = false;
    let fille_name;
    let total_frame;
    let low_frame_num;
    let upload_state;
    let detecting_high_state;
    let detecting_low_state;
    let is_play = false;
    let trigger_stop = false;

    const safeClasses = ["road","hiking_trail", "restarea", "parking_lot", "car"];

    $(document).ready(function() {

        /* SCROLLING NAVBAR */
        var OFFSET_TOP = 50;

        $(window).scroll(function () {
            if ($('.navbar').length) {
                if ($('.navbar').offset().top > OFFSET_TOP) {
                    $('.scrolling-navbar').addClass('top-nav-collapse');
                } else {
                    $('.scrolling-navbar').removeClass('top-nav-collapse');
                }
            }
        });

        // file uploading 
        let fileDropZone = new Dropzone("div#dZUpload", {
            url: '/upload',
            paramName: 'file',
            autoProcessQueue: false,
            chunking: true,
            forceChunking: true,
            maxFilesize: 1024 * 10, // megabytes
            chunkSize: 1024 * 1024 * 10, // bytes
            acceptedFiles: ".mp4",
            init: function() {
                var myDropzone = this;
                $('#btn_upload').click(function() {
                    myDropzone.processQueue();
                });

                this.on("success", function(file, response) {
                    getState();
                    startGetStateLoop();
                });

                this.on("error", function(a, b) {
                    console.log("error");
                    console.log(a);
                    console.log(b);
                })
            }
        });

        //clear Session
        $("#clearSession").click(function(event) {
            event.preventDefault();
            $.ajax({
                url: '/clearSession',
                success: function(data) {
                    console.log(data);
                    location.reload();
                }
            });
        });

        $("#downloadBtn").click(function(event){
            event.preventDefault();
            console.log($(this).data('state'), 'log');
            if(detecting_high_state == 'finished'){
                location.href='/result_file_download';
            }
        });

        $("#playAgain").click(function(event){
            event.preventDefault();
            if(detecting_low_state == 'finished'){
                if(!is_play){
                    $("#playAgain").html("STOP");
                    trigger_stop = false;
                    playVideo_low_frame(0, low_frame_num);
                }else{
                    trigger_stop = true;
                }
            }
        });

        $("#progressBar").change(function(){
            if(is_play) return;
            if(detecting_low_state != "finished") return;
            frame_number = $(this).val() - 1;
            if(frame_number < 0) frame_number = 0;
                
            $.ajax({
                url: '/getImage/'+frame_number,
                success: function(data) {
                    $("#frameImg").attr('src', data.src);
                    displayResult(data.result);
                }
            });
        });

        // $('#btn_playFrame').click(function(event) {
        //     event.preventDefault();
        //     let frame_num = $('#in_frame_num').val()
        //     // $('#stream_img').attr('src', '/detecting/' + frame_num);
        //     $.ajax({
        //         url: '/detecting/' + frame_num,
        //         success: function(data) {
        //             console.log(data);
        //             //$('#stream_img').attr('src', "data:image/jpg;base64," + data);
        //             $('#stream_img').attr('src', data);
        //         }
        //     });
        // });

        // $("#btn_playAgain").click(function(event) {
        //     event.preventDefault();
        //     $('#video_stream_area').show();
        //     $('#stream_img').attr('src', '/stream_detection');
        // });

        
        getState();
    });

    let stateInterval;
    function startGetStateLoop(){
        if(detecting_high_state != 'finished' || detecting_low_state != 'finished'){
            stateInterval = setInterval(getState, 3000);
        }
    }

    function getState() {
        $.ajax({
            url: '/getState',
            success: function(data) {
                console.log(data)
                fille_name = data.fille_name;
                total_frame = data.total_frame;
                upload_state = data.upload_state;
                detecting_high_state = data.detecting_high_state;
                detecting_low_state = data.detecting_low_state;
                low_frame_num = data.low_frame_num;
                current_detected_file = data.current_detected_file;
                $("#playAgain").data('state', detecting_low_state);
                $("#downloadBtn").data('state', detecting_high_state);
                if(detecting_high_state == 'finished' && detecting_low_state == 'finished'){
                    clearInterval(stateInterval);
                }

                if (upload_state) {
                    $("#uploadArea").hide();

                    switch (detecting_low_state) {
                        case 'finished':
                            if(!is_play){
                                $("#playAgain").html("PLAY");
                            }
                            // $('#progressBar').attr('aria-valuemax', low_frame_num);
                            $('#progressBar').css('display', 'block');
                            $('#progressBar').attr('min', 0).attr('max', low_frame_num);
                            break;
                        case 'ongoing':
                            $("#playAgain").html("변환중");
                            $('#progressBar').css('display', 'none');
                            if( current_detected_file != null){
                                $("#frameImg").attr('src', current_detected_file);
                                displayResult(data.result);
                            }
                            break;
                        case 'idle':
                            $("#playAgain").html("변환전");
                            $('#progressBar').css('display', 'none');
                            break;
                    }
                    switch (detecting_high_state) {
                        case 'finished':
                            $("#downloadBtn").html("다운로드");
                            break;
                        case 'ongoing':
                            $("#downloadBtn").html("변환중");
                            break;
                        case 'idle':
                            $("#downloadBtn").html("변환전");
                            break;
                    }

                    $("#displayArea").css('display', 'flex');
                } else {
                    $("#displayArea").hide();
                    $("#playAgain").html('변환전');
                    $("#playAgain").html('변환전');
                    $("#playAgain").html('변환전');
                    $("#uploadArea").css('display', 'flex');
                }
            }
        });
    }

    function playVideo_low_frame(current_frame, low_frame_num){
        is_play = true;
        if(current_frame >= low_frame_num || trigger_stop){
            is_play = false;
            $("#playAgain").html("PLAY");
            return;
        }
        now_percent = (current_frame + 1) * 100 / low_frame_num;
        $.ajax({
            url: '/getImage/'+current_frame,
            success: function(data) {
                // console.log(data);
                
                $("#frameImg").attr('src', data.src);
                displayResult(data.result);
                // $('#progressBar').css('width', now_percent+'%').attr('aria-valuenow', current_frame);
                //$('#frameImg').attr('src', "data:image/jpg;base64," + data);
                $('#progressBar').val(current_frame+1);
            }
        });
        setTimeout(function(){
            playVideo_low_frame(current_frame+1, low_frame_num);
        }, 1000);
    }


    function displayResult(result){
        $("#totalPsersonCount").text("Count of detected people : "+result.person_count);
        let detailList = $('#personList');
        detailList.children().remove();
        if(result.person_count > 0){
            $.each(result.detect_result, function(idx, elm){
                let itemState = "list-group-item-light";
                let strIntersection = "";
                if(elm.className == "person_ab") {
                    itemState = "list-group-item-danger";
                }else if(elm.isIntersect == false){
                     itemState = "list-group-item-warning";
                }else{
                    strIntersection = 'on the '+elm.intersectedClassName;
                    if($.inArray(elm.intersectedClassName, safeClasses)){
                        itemState = "list-group-item-info";
                    }else{
                        itemState = "list-group-item-danger";
                    }
                }
                detailList.append('<li class="list-group-item '+itemState+'">'+elm.className+' '+strIntersection+'</li>');
            });
        }
    }

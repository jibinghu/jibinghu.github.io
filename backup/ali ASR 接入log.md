```
2024-12-11 12:31:05.982418 98.23% [DEBUG] ali_asr_ws.c:691 ali_asr license used: 1
2024-12-11 12:31:05.982418 98.23% [INFO] ali_asr_ws.c:715 codec = L16, rate = 8000, dest = (null)
0193b3fb-b7e1-7969-962c-f95396b9994e 2024-12-11 12:31:05.982418 98.23% [DEBUG] switch_core_media_bug.c:1003 Attaching BUG detect_speech to sofia/public/15513689240
2024-12-11 12:31:05.982418 98.23% [NOTICE] ali_asr_ws.c:467 WS read thread start
2024-12-11 12:31:05.982418 98.23% [DEBUG] ali_asr_ws.c:474 Connecting to API WS socket wss://nls-gateway.cn-shanghai.aliyuncs.com/ws/v1
0193b3fb-b7e1-7969-962c-f95396b9994e 2024-12-11 12:31:05.982418 98.23% [INFO] billing.c:507 detect_speech media bug started
2024-12-11 12:31:05.982418 98.23% [DEBUG] ali_asr_ws.c:1128 start-input-timers = 1
2024-12-11 12:31:05.982418 98.23% [DEBUG] ali_asr_ws.c:1111 no-input-timeout = 5000
2024-12-11 12:31:05.982418 98.23% [DEBUG] ali_asr_ws.c:1119 speech-timeout = 15000
2024-12-11 12:31:05.982418 98.23% [DEBUG] ali_asr_ws.c:1166 add-punc = true
2024-12-11 12:31:05.982418 98.23% [DEBUG] ali_asr_ws.c:1236 enable-inverse-text-normalization = true
2024-12-11 12:31:05.982418 98.23% [INFO] ali_asr_ws.c:774 load grammar default
0193b3fb-b7e1-7969-962c-f95396b9994e 2024-12-11 12:31:05.982418 98.23% [DEBUG] switch_ivr_play_say.c:1561 Codec Activated L16@8000hz 1 channels 20ms
2024-12-11 12:31:06.222415 98.20% [INFO] ali_asr_ws.c:351 Websocket connected to [wss://nls-gateway.cn-shanghai.aliyuncs.com/ws/v1]
2024-12-11 12:31:06.222415 98.20% [INFO] ali_asr_ws.c:275 {"header":{"namespace":"SpeechTranscriber","name":"StartTranscription","task_id":"a426f3d4618447519c9d85d1a0d15bf6","message_id":"a426f3d4618447519c9d85d1a0d15bf6","appkey":"InIBfoN6rSWJhaOW"},"payload":{"format":"pcm","sample_rate":8000,"max_sentence_silence":500,"enable_punctuation_prediction":true,"enable_inverse_text_normalization":true},"context":{"sdk":{"name":"xswitch-nls-sdk-c","version":"1.0.0"}}}
0193b3fb-b7e1-7969-962c-f95396b9994e 2024-12-11 12:31:06.242417 98.20% [DEBUG] switch_core_io.c:448 Setting BUG Codec PCMA:8
2024-12-11 12:31:06.282416 98.20% [INFO] ali_asr_ws.c:522 {"header":{"namespace":"SpeechTranscriber","name":"TranscriptionStarted","status":20000000,"message_id":"acaeaef171f3411497c5f6333d0e35a3","task_id":"a426f3d4618447519c9d85d1a0d15bf6","status_text":"Gateway:SUCCESS:Success."}}
2024-12-11 12:31:06.282416 98.20% [INFO] ali_asr_ws.c:534 TranscriptionStarted
0193b3fb-b7e1-7969-962c-f95396b9994e 2024-12-11 12:31:07.202414 98.20% [DEBUG] switch_ivr_play_say.c:2012 done playing file silence_stream://1000
2024-12-11 12:31:07.202414 98.20% [INFO] ali_asr_ws.c:1094 Input timers already started
0193b3fb-b7e1-7969-962c-f95396b9994e 2024-12-11 12:31:07.202414 98.20% [INFO] switch_ivr_async.c:5014 (sofia/public/15513689240) WAITING FOR RESULT
0193b3fb-b7e1-7969-962c-f95396b9994e 2024-12-11 12:31:07.202414 98.20% [DEBUG] switch_ivr.c:195 Codec Activated L16@8000hz 1 channels 20ms
2024-12-11 12:31:07.522428 98.20% [INFO] ali_asr_ws.c:522 {"header":{"namespace":"SpeechTranscriber","name":"SentenceBegin","status":20000000,"message_id":"a13068a127fe42bc955c8f203a088f35","task_id":"a426f3d4618447519c9d85d1a0d15bf6","status_text":"Gateway:SUCCESS:Success."},"payload":{"index":1,"time":940}}
2024-12-11 12:31:07.522428 98.20% [INFO] ali_asr_ws.c:545 SentenceBegin
2024-12-11 12:31:07.542429 98.20% [DEBUG] ali_asr_ws.c:1059 Result: START OF SPEECH
0193b3fb-b7e1-7969-962c-f95396b9994e 2024-12-11 12:31:07.562425 98.20% [INFO] switch_ivr_async.c:4918 (sofia/public/15513689240) START OF SPEECH
0193b3fb-b7e1-7969-962c-f95396b9994e 2024-12-11 12:31:07.562425 98.20% [DEBUG] switch_ivr.c:195 Codec Activated L16@8000hz 1 channels 20ms
2024-12-11 12:31:09.222436 98.23% [INFO] ali_asr_ws.c:522 {"header":{"namespace":"SpeechTranscriber","name":"SentenceEnd","status":20000000,"message_id":"4baa38c45e02462d81c9b4e0db2260ae","task_id":"a426f3d4618447519c9d85d1a0d15bf6","status_text":"Gateway:SUCCESS:Success."},"payload":{"index":1,"time":2620,"result":"是的是的是的是的。","confidence":0.858,"words":[],"status":0,"gender":"","begin_time":940,"fixed_result":"","unfixed_result":"","stash_result":{"sentenceId":2,"beginTime":2620,"text":"","fixedText":"","unfixedText":"","currentTime":2620,"words":[]},"audio_extra_info":"","sentence_id":"aef538e13b0449139ea803ef7b9c9297","gender_score":0.0}}
2024-12-11 12:31:09.222436 98.23% [INFO] ali_asr_ws.c:573 SentenceEnd
2024-12-11 12:31:09.222436 98.23% [NOTICE] ali_asr_ws.c:1043 Recognized: {"engine":"ali","text":"是的是的是的是的。","confidence":0.858,"engine_data":{"header":{"namespace":"SpeechTranscriber","name":"SentenceEnd","status":20000000,"message_id":"4baa38c45e02462d81c9b4e0db2260ae","task_id":"a426f3d4618447519c9d85d1a0d15bf6","status_text":"Gateway:SUCCESS:Success."},"payload":{"index":1,"time":2620,"result":"是的是的是的是的。","confidence":0.858,"words":[],"status":0,"gender":"","begin_time":940,"fixed_result":"","unfixed_result":"","stash_result":{"sentenceId":2,"beginTime":2620,"text":"","fixedText":"","unfixedText":"","currentTime":2620,"words":[]},"audio_extra_info":"","sentence_id":"aef538e13b0449139ea803ef7b9c9297","gender_score":0}}}, Confidence: 0.858000
0193b3fb-b7e1-7969-962c-f95396b9994e 2024-12-11 12:31:09.242419 98.23% [INFO] switch_ivr_async.c:4905 (sofia/public/15513689240) DETECTED SPEECH
2024-12-11 12:31:09.242419 98.23% [ERR] ai_app.c:900 speach
2024-12-11 12:31:09.242419 98.23% [ERR] ai_app.c:904 detected: {"engine":"ali","text":"是的是的是的是的。","confidence":0.858,"engine_data":{"header":{"namespace":"SpeechTranscriber","name":"SentenceEnd","status":20000000,"message_id":"4baa38c45e02462d81c9b4e0db2260ae","task_id":"a426f3d4618447519c9d85d1a0d15bf6","status_text":"Gateway:SUCCESS:Success."},"payload":{"index":1,"time":2620,"result":"是的是的是的是的。","confidence":0.858,"words":[],"status":0,"gender":"","begin_time":940,"fixed_result":"","unfixed_result":"","stash_result":{"sentenceId":2,"beginTime":2620,"text":"","fixedText":"","unfixedText":"","currentTime":2620,"words":[]},"audio_extra_info":"","sentence_id":"aef538e13b0449139ea803ef7b9c9297","gender_score":0}}}
2024-12-11 12:31:09.242419 98.23% [ERR] ai_app.c:911 {
        "engine":       "ali",
        "text": "是的是的是的是的。",
        "confidence":   0.858,
        "engine_data":  {
                "header":       {
                        "namespace":    "SpeechTranscriber",
                        "name": "SentenceEnd",
                        "status":       20000000,
                        "message_id":   "4baa38c45e02462d81c9b4e0db2260ae",
                        "task_id":      "a426f3d4618447519c9d85d1a0d15bf6",
                        "status_text":  "Gateway:SUCCESS:Success."
                },
                "payload":      {
                        "index":        1,
                        "time": 2620,
                        "result":       "是的是的是的是的。",
                        "confidence":   0.858,
                        "words":        [],
                        "status":       0,
                        "gender":       "",
                        "begin_time":   940,
                        "fixed_result": "",
                        "unfixed_result":       "",
                        "stash_result": {
                                "sentenceId":   2,
                                "beginTime":    2620,
                                "text": "",
                                "fixedText":    "",
                                "unfixedText":  "",
                                "currentTime":  2620,
                                "words":        []
                        },
                        "audio_extra_info":     "",
                        "sentence_id":  "aef538e13b0449139ea803ef7b9c9297",
                        "gender_score": 0
                }
        }
}
2024-12-11 12:31:09.242419 98.23% [DEBUG] mod_ai.c:243 sending request to: http://36.212.25.245:9393/outbound_result
request_data: {"uuid":"0193b3fb-b7e1-7969-962c-f95396b9994e","project":"taihang","machine_id":"5058","hostname":"wrzh001","cid_number":"16696128104","dest_number":"15513689240","call_status":"ACTIVE","direction":"outbound","url":"http://36.212.25.245:9393/outbound_result","name":"dialed_digits","input_type":"detected_speech","asr_result":{"engine":"ali","text":"是的是的是的是的。","confidence":0.858,"engine_data":{"header":{"namespace":"SpeechTranscriber","name":"SentenceEnd","status":20000000,"message_id":"4baa38c45e02462d81c9b4e0db2260ae","task_id":"a426f3d4618447519c9d85d1a0d15bf6","status_text":"Gateway:SUCCESS:Success."},"payload":{"index":1,"time":2620,"result":"是的是的是的是的。","confidence":0.858,"words":[],"status":0,"gender":"","begin_time":940,"fixed_result":"","unfixed_result":"","stash_result":{"sentenceId":2,"beginTime":2620,"text":"","fixedText":"","unfixedText":"","currentTime":2620,"words":[]},"audio_extra_info":"","sentence_id":"aef538e13b0449139ea803ef7b9c9297","gender_score":0}}},"play_finished":0,"private_data":{"data1":"a","data2":2},"local_ip_v4":"172.31.0.5","channel_data":{"Caller-Direction":"outbound","Caller-Logical-Direction":"outbound","Caller-Caller-ID-Name":"yxt16696128104","Caller-Caller-ID-Number":"yxt16696128104","Caller-Orig-Caller-ID-Name":"16696128104","Caller-Orig-Caller-ID-Number":"16696128104","Caller-Callee-ID-Name":"16696128104","Caller-Callee-ID-Number":"16696128104","Caller-Network-Addr":"47.94.86.132","Caller-ANI":"16696128104","Caller-Destination-Number":"15513689240","Caller-Unique-ID":"0193b3fb-b7e1-7969-962c-f95396b9994e","Caller-Source":"src/switch_ivr_originate.c","Caller-Context":"default","Caller-Channel-Name":"sofia/public/15513689240","Caller-Profile-Index":"1","Caller-Profile-Created-Time":"1733891438542418","Caller-Channel-Created-Time":"1733891438542418","Caller-Channel-Answered-Time":"1733891453322424","Caller-Channel-Progress-Time":"0","Caller-Channel-Progress-Media-Time":"1733891438602418","Caller-Channel-Hangup-Time":"0","Caller-Channel-Transfer-Time":"0","Caller-Channel-Resurrect-Time":"0","Caller-Channel-Bridged-Time":"0","Caller-Channel-Last-Hold":"0","Caller-Channel-Hold-Accum":"0","Caller-Screen-Bit":"true","Caller-Privacy-Hide-Name":"false","Caller-Privacy-Hide-Number":"false"},"max_sessions":100,"session_count":1,"sps":30}
```
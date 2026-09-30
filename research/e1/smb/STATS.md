# SMB v0 — stats

Generated 2026-09-17. QC: 300/300 queries verified against raw label files, 0 errors (qc_check.py).

~~~
== inventory stats ==
{
 "ucf": {
  "n_videos": 290,
  "n_anomaly_videos": 140,
  "n_events": 156,
  "total_hours": 10.28,
  "event_duration_mean_s": 18.5,
  "category_event_counts": {
   "Shooting": 25,
   "Shoplifting": 25,
   "RoadAccidents": 23,
   "Explosion": 22,
   "Burglary": 15,
   "Arson": 10,
   "Vandalism": 8,
   "Stealing": 7,
   "Arrest": 5,
   "Fighting": 5,
   "Robbery": 5,
   "Assault": 4,
   "Abuse": 2
  },
  "cameras": []
 },
 "xd": {
  "n_videos": 800,
  "n_anomaly_videos": 500,
  "n_events": 1238,
  "total_hours": 26.97,
  "event_duration_mean_s": 18.7,
  "category_event_counts": {
   "Car Accident": 513,
   "Fighting": 275,
   "Shooting": 261,
   "Explosion": 192,
   "Riot": 157,
   "Abuse": 19
  },
  "cameras": []
 },
 "sht": {
  "n_videos": 107,
  "n_anomaly_videos": 107,
  "n_events": 194,
  "total_hours": 0.45,
  "event_duration_mean_s": 3.6,
  "category_event_counts": {
   "Anomaly": 194
  },
  "cameras": [
   "01",
   "02",
   "03",
   "04",
   "05",
   "06",
   "07",
   "08",
   "09",
   "10",
   "11",
   "12"
  ]
 }
}

== queries per type x dataset ==
type               ucf      xd     sht   TOTAL
existence           30      30      15      75
temporal            30      30      15      75
retrieval           25      25      10      60
cross_camera         0       0      30      30
negation            25      25      10      60

== difficulty ==
Counter({'corpus': 60, 'absent_cat': 60, 'long_highpeak': 51, 'short_highpeak': 37, 'fleet': 30, 'short': 29, 'short_lowpeak': 16, 'long_lowpeak': 16, 'long': 1})

== category distribution in queries (existence+temporal+negation) ==
{'Shooting': 27, 'Explosion': 26, 'Abuse': 18, 'Riot': 15, 'Car Accident': 15, 'Arson': 10, 'Burglary': 8, 'Robbery': 8, 'Shoplifting': 8, 'Assault': 7, 'Arrest': 6, 'Fighting': 4, 'RoadAccidents': 4, 'Stealing': 4, 'Vandalism': 2}

== paraphrase usage ==
LLM-paraphrased: 238/300

== 15 random queries (verbatim) ==
[q0165 ucf/retrieval] Search for videos with the arrest event type.
    answer: {"relevant_videos": ["Arrest001_x264", "Arrest007_x264", "Arrest024_x264", "Arrest030_x264", "Arrest039_x264"], "n_relevant": 5}
[q0077 ucf/temporal] When did the arson incident occur in video Arson007_x264?
    answer: {"spans": [[74.67, 190.4]]}
[q0202 sht/retrieval] Find all videos from camera 03 that contain an anomalous event.
    answer: {"relevant_videos": ["03_0031", "03_0032", "03_0033", "03_0035", "03_0036", "03_0039", "03_0041", "03_0059", "03_0060", "03_0061"], "n_relevant": 10}
[q0024 ucf/existence] Did the theft occur in video Stealing058_x264?
    answer: {"value": "yes"}
[q0037 xd/existence] Did the video v=rKpXqwE2rg8__#00-01-57_00-03-06_label_B4-0-0 capture any signs of a riot?
    answer: {"value": "yes"}
[q0274 xd/negation] Did the video Bad.Boys.II.2003__#01-11-16_01-14-00_label_A capture any fights?
    answer: {"value": "no"}
[q0048 xd/existence] Was there a shooting captured in video Bad.Boys.1995__#01-33-51_01-34-37_label_B2-0-0?
    answer: {"value": "yes"}
[q0187 xd/retrieval] Locate videos where an explosion is captured.
    answer: {"relevant_videos": ["Bad.Boys.1995__#01-11-55_01-12-40_label_G-B2-B6", "Bad.Boys.II.2003__#00-06-42_00-10-00_label_B2-G-0", "Black.Hawk.Down.2001__#01-42-58_01
[q0298 sht/negation] Did the footage from camera 01 in video 01_0053 capture a shooting incident?
    answer: {"value": "no"}
[q0029 ucf/existence] Did the video Burglary061_x264 show a burglary?
    answer: {"value": "yes"}
[q0259 ucf/negation] Did a fight occur in the video Normal_Videos_641_x264?
    answer: {"value": "no"}
[q0109 xd/temporal] What is the timestamp of the explosion captured in video v=qrKfaX1lCUM__#1_label_G-0-0?
    answer: {"spans": [[66.67, 69.33]]}
[q0019 ucf/existence] Was there any fighting activity captured in video ID Fighting033_x264?
    answer: {"value": "yes"}
[q0044 xd/existence] Was there an explosion in video v=v_LxqgpRouM__#1_label_G-0-0?
    answer: {"value": "yes"}
[q0222 sht/cross_camera] Which camera recorded the largest number of anomalous events?
    answer: {"camera": "01", "n_events": 45}
~~~

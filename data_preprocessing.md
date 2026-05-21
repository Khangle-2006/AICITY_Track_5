# Data-preprocessing

**Scenario** là một tình huống/góc quay sự kiện lớn. Một scenario có thể có nhiều video view.

- Main WTS và Normal Trimmed **không nằm ở hai folder top-level riêng**, mà cùng nằm trong video/train, video/val, caption/train, caption/val. Mình phân biệt chúng bằng tên folder: folder có normal là normal_trimmed, còn các folder như 20230707_12_SN17_T1 là Main WTS.
- Không phải folder nào cũng có vehicle_view. Vì vậy số vehicle JSON/video thấp hơn overhead JSON/video.

```jsx
WTS_dataset/
├── video/
│   ├── train/
│   │   ├── 20230707_12_SN17_T1/                 # Main WTS scenario
│   │   │   ├── overhead_view/                   # Multiple fixed/overhead cameras
│   │   │   │   ├── 20230707_12_SN17_T1_Camera1_0.mp4
│   │   │   │   ├── 20230707_12_SN17_T1_Camera2_3.mp4
│   │   │   │   ├── 20230707_12_SN17_T1_Camera3_1.mp4
│   │   │   │   └── 20230707_12_SN17_T1_Camera4_2.mp4
│   │   │   └── vehicle_view/                    # Vehicle/ego-view camera, if available
│   │   │       └── 20230707_12_SN17_T1_vehicle_view.mp4
│   │   │
│   │   ├── 20230707_15_SY4_T1/                 # Another Main WTS scenario
│   │   │   ├── overhead_view/
│   │   │   │   ├── 20230707_15_SY4_T1_Camera1_0.mp4
│   │   │   │   ├── 20230707_15_SY4_T1_Camera2_1.mp4
│   │   │   │   └── 20230707_15_SY4_T1_Camera3_2.mp4
│   │   │   └── vehicle_view/
│   │   │       └── 20230707_15_SY4_T1_vehicle_view.mp4
│   │   │
│   │   ├── 20231013_101813_normal_192.168.0.11_1_event_1/
│   │   │   ├── overhead_view/                   # Normal trimmed WTS sample
│   │   │   │   └── ...mp4
│   │   │   └── vehicle_view/                    # May be missing for some folders
│   │   │       └── ...mp4
│   │   │
│   │   └── ...
│   │
│   └── val/
│       ├── 20230728_28_SN20_T1/                # Main WTS validation scenario
│       │   ├── overhead_view/
│       │   │   └── ...mp4
│       │   └── vehicle_view/                    # May be missing
│       │       └── ...mp4
│       │
│       ├── 20231013_105827_normal_192.168.0.13_4_event_0/
│       │   ├── overhead_view/                   # Normal trimmed validation sample
│       │   │   └── ...mp4
│       │   └── vehicle_view/                    # May be missing
│       │       └── ...mp4
│       │
│       └── ...
│
├── caption/
│   ├── train/
│   │   ├── 20230707_12_SN17_T1/
│   │   │   ├── overhead_view/
│   │   │   │   └── 20230707_12_SN17_T1_caption.json
│   │   │   └── vehicle_view/
│   │   │       └── 20230707_12_SN17_T1_caption.json
│   │   │
│   │   ├── 20231013_101813_normal_192.168.0.11_1_event_1/
│   │   │   ├── overhead_view/
│   │   │   │   └── ..._caption.json
│   │   │   └── vehicle_view/                    # May be missing
│   │   │       └── ..._caption.json
│   │   │
│   │   └── ...
│   │
│   └── val/
│       ├── 20230728_28_SN20_T1/
│       │   ├── overhead_view/
│       │   │   └── ..._caption.json
│       │   └── vehicle_view/
│       │       └── ..._caption.json
│       │
│       ├── 20231013_105827_normal_192.168.0.13_4_event_0/
│       │   ├── overhead_view/
│       │   │   └── ..._caption.json
│       │   └── vehicle_view/                    # May be missing
│       │       └── ..._caption.json
│       │
│       └── ...
│
└── external/
    ├── video/
    │   ├── train/
    │   │   ├── video1004.mp4
    │   │   ├── video1006.mp4
    │   │   ├── video1009.mp4
    │   │   └── ...
    │   │
    │   └── val/
    │       ├── video135.mp4
    │       ├── video225.mp4
    │       └── ...
    │
    └── caption/
        ├── train/
        │   ├── video1004_caption.json
        │   ├── video1006_caption.json
        │   ├── video1009_caption.json
        │   └── ...
        │
        └── val/
            ├── video135_caption.json
            ├── video225_caption.json
            └── ...
```

```jsx
WTS_TRACK5_TEST/
└── WTS_TRACK5_TEST/
    ├── 20230707_14_CN16_T1_Camera2_3/              # WTS overhead/fixed camera sample
    │   ├── input/                                  # History frames
    │   │   ├── 0.png
    │   │   ├── 1.png
    │   │   ├── 2.png
    │   │   └── ...
    │   └── caption.json                            # One future phase caption + frame length
    │
    ├── 20231006_18_CN29_T1_192.168.0.11_1/         # WTS vehicle/IP camera sample
    │   ├── input/
    │   │   ├── 0.png
    │   │   ├── 1.png
    │   │   ├── 2.png
    │   │   └── ...
    │   └── caption.json
    │
    ├── 20231013_101824_normal_192.168.0.13_4_event_0/
    │   ├── input/                                  # WTS normal_trimmed vehicle/IP-style sample
    │   │   ├── 0.png
    │   │   ├── 1.png
    │   │   ├── 2.png
    │   │   └── ...
    │   └── caption.json
    │
    ├── video135/                                   # BDD_PC_5K sample
    │   ├── input/
    │   │   ├── 0.png
    │   │   ├── 1.png
    │   │   ├── 2.png
    │   │   └── ...
    │   └── caption.json
    │
    ├── video225/                                   # Another BDD_PC_5K sample
    │   ├── input/
    │   │   ├── 0.png
    │   │   ├── 1.png
    │   │   ├── 2.png
    │   │   └── ...
    │   └── caption.json
    │
    └── ...

```

**Segment row** là một đoạn nhỏ bên trong caption.json.

Labels: 

0: pre-recognition
1: recognition
2: judgment
3: action
4: avoidance

**Vehicle view:** 

```json
{
    "vehicle_view": "20230707_12_SN17_T1_vehicle_view.mp4",
    "event_phase": [
        {
            "labels": [
                "4"
            ],
            "caption_pedestrian": "The pedestrian, a male in his 30s approximately 170 cm tall, was wearing a black T-shirt and black slacks. It was a clear and bright day with dry road conditions on a residential road with two-way traffic. There were no sidewalks on both sides, and street lights were present. The pedestrian was positioned directly in front of a vehicle, facing the opposite direction. The pedestrian noticed the vehicle and was slowly moving in front of it. Suddenly, a collision occurred.",
            "caption_vehicle": "The vehicle is positioned in front of a pedestrian, close in proximity. The vehicle has a clear field of view, as the pedestrian is visible. The vehicle is currently stopped and its speed is 0 km/h. The gender of the pedestrian is male, in his 30s with a height of 170 cm. He is wearing a black T-shirt on the upper body and black slacks on the lower body. The weather is clear and the brightness is bright. The road surface conditions are dry and the road is level with asphalt. The traffic volume is usual on this two-way residential road. There is no sidewalk on both sides, and both the roadside strip and street lights are present.",
            "start_time": "9.476",
            "end_time": "14.017"
        },
        {
            "labels": [
                "3"
            ],
            "caption_pedestrian": "The pedestrian, a male in his 30s, stood perpendicular to the vehicle and to the right of it. He was directly in front of the vehicle, positioned closely. His line of sight was focused on the road surface, indicating he was closely watching. Moving slowly, the pedestrian was in front of the vehicle and appeared to be crossing. He wore a black T-shirt on his upper body and black slacks on his lower body. The weather was clear with bright brightness. The road surface was dry and leveled asphalt. The traffic volume on the two-way residential road was usual. Both sides of the road did not have a sidewalk or roadside strip, but there were street lights. This set of data provides a detailed description of the pedestrian and the surrounding environment in a third-person narrative style.",
            "caption_vehicle": "The vehicle is positioned on the right side of the pedestrian and is in close proximity to them. From the vehicle's field of view, the pedestrian is visible. The vehicle is going straight ahead at a speed of 5 km/h. The environment conditions include a male pedestrian in his 30s, standing at a height of 170 cm. He is wearing a black T-shirt for his upper body and black slacks for his lower body. The weather is clear and the brightness is bright. The road surface conditions are dry and the road is level with asphalt. The traffic volume is usual on this residential road with two-way traffic. There is no sidewalk available on both sides, and the roadside strip is also absent. Street lights are present in the surroundings.",
            "start_time": "5.383",
            "end_time": "9.420"
        },
        {
            "labels": [
                "2"
            ],
            "caption_pedestrian": "A man in his 30s wearing a black T-shirt and black slacks stands diagonally to the left of a vehicle on a residential road. His body is positioned perpendicular to the vehicle and to the right. The man's height is approximately 170 cm. The weather is clear, and the brightness is bright. The road surface is dry and level, made of asphalt. There are two-way traffic lanes, but there is no sidewalk or roadside strip on both sides of the road. Street lights illuminate the surroundings. Despite noticing the vehicle, the pedestrian, moving slowly, seems to be slowly looking around, possibly unaware of the vehicle's presence. The pedestrian is about to cross the road, intending to travel in front of the vehicle. The traffic volume is usual, creating a typical scenario for the pedestrian.",
            "caption_vehicle": "The vehicle is positioned diagonally to the right in front of the pedestrian and is at a close distance from them. The pedestrian is visible within the vehicle's field of view. The vehicle is currently stopped and its speed is 0 km/h. The driver is observing the surroundings and waiting for the appropriate moment to continue. Meanwhile, the environment conditions indicate that the pedestrian is a male in his 30s, approximately 170 cm tall, wearing a black T-shirt and black slacks. The weather is clear and the brightness is bright. The road conditions are favorable, with a dry asphalt surface and a level incline. The road is a residential road with two-way traffic and without sidewalks on both sides or roadside strips on both sides. However, there are street lights illuminating the area. The overall situation seems calm and ordinary, with the vehicle and pedestrian being momentarily paused in their respective positions on the road.",
            "start_time": "4.093",
            "end_time": "5.383"
        },
        {
            "labels": [
                "1"
            ],
            "caption_pedestrian": "The pedestrian, a male in his 30s, stood diagonally to the left in front of the vehicle. His body was perpendicular to the vehicle and to the right. He was approximately close to the vehicle, with a clear line of sight towards it. Slowly looking around, he was aware of the vehicle's presence. Dressed in a black T-shirt and slacks, he stood on a level and dry asphalt road. The weather was clear with bright lighting, and it was a usual day in terms of traffic volume. The event took place on a residential road with two-way traffic. There was no sidewalk on both sides, and neither the roadside strip nor the street lights were present. Despite this, the pedestrian was about to cross the road, moving slowly in the direction of travel in front of the vehicle.",
            "caption_vehicle": "The vehicle is positioned diagonally to the right in front of the pedestrian, at a close relative distance. The pedestrian is clearly visible within the vehicle's field of view. The vehicle is about to stop and is moving at a speed of 5 km/h. The driver is aware of the pedestrian's presence and is taking necessary action to halt the vehicle. The event takes place in a clear weather condition with bright brightness. The vehicle is traveling on a residential road with two-way traffic and a dry asphalt surface. The surrounding environment indicates a male pedestrian in his 30s, standing at a height of 170 cm. He is wearing a black T-shirt and black slacks. The road does not have a sidewalk on both sides and there is no roadside strip. However, street lights are present, ensuring sufficient visibility. The traffic volume is normal and the road is level. All these factors provide the necessary context for the event involving the vehicle and the pedestrian.",
            "start_time": "1.066",
            "end_time": "4.037"
        },
        {
            "labels": [
                "0"
            ],
            "caption_pedestrian": "The pedestrian, a male in his 30s, was standing diagonally to the left in front of the vehicle on a residential road. His body was oriented perpendicular to the vehicle and to the right. Positioned near the vehicle, his line of sight was directed towards his crossing destination. He was slowly looking around and dressed in a black T-shirt and slacks. The weather was clear and bright, with dry road surface conditions on the level asphalt road. Despite the usual traffic volume on the two-way street, there were no sidewalks or roadside strips available. However, street lights were present, ensuring visibility. The pedestrian's general action was going straight ahead, and his speed was slow. Overall, the pedestrian was actively observing his surroundings and preparing to cross the road, taking into account the environmental conditions and traffic situation.",
            "caption_vehicle": "The vehicle is positioned diagonally to the right in front of the pedestrian, at a close distance. The pedestrian is visible within the vehicle's field of view. The vehicle is going straight ahead at a speed of 10 km/h. Meanwhile, in the environment, there is a male pedestrian in his 30s, standing at a height of 170 cm. He is wearing a black T-shirt and black slacks. The weather is clear with bright lighting, and the road surface is dry and level. The road is a residential road with two-way traffic and does not have sidewalks on both sides. There are no roadside strips, but the street lights are on. Overall, the vehicle is in a normal traffic situation, with clear visibility of the pedestrian and suitable road conditions for its speed and direction.",
            "start_time": "0.00",
            "end_time": "1.066"
        }
    ]
}
```

**Overhear view:** 

```json
{
    "id": 726,
    "overhead_videos": [
        "20230707_12_SN17_T1_Camera1_0.mp4",
        "20230707_12_SN17_T1_Camera3_1.mp4",
        "20230707_12_SN17_T1_Camera4_2.mp4",
        "20230707_12_SN17_T1_Camera2_3.mp4"
    ],
    "event_phase": [
        {
            "labels": [
                "4"
            ],
            "caption_pedestrian": "The pedestrian, a male in his 30s approximately 170 cm tall, was wearing a black T-shirt and black slacks. It was a clear and bright day with dry road conditions on a residential road with two-way traffic. There were no sidewalks on both sides, and street lights were present. The pedestrian was positioned directly in front of a vehicle, facing the opposite direction. The pedestrian noticed the vehicle and was slowly moving in front of it. Suddenly, a collision occurred.",
            "caption_vehicle": "The vehicle is positioned in front of a pedestrian, close in proximity. The vehicle has a clear field of view, as the pedestrian is visible. The vehicle is currently stopped and its speed is 0 km/h. The gender of the pedestrian is male, in his 30s with a height of 170 cm. He is wearing a black T-shirt on the upper body and black slacks on the lower body. The weather is clear and the brightness is bright. The road surface conditions are dry and the road is level with asphalt. The traffic volume is usual on this two-way residential road. There is no sidewalk on both sides, and both the roadside strip and street lights are present.",
            "start_time": "37.734",
            "end_time": "42.275"
        },
        {
            "labels": [
                "3"
            ],
            "caption_pedestrian": "The pedestrian, a male in his 30s, stood perpendicular to the vehicle and to the right of it. He was directly in front of the vehicle, positioned closely. His line of sight was focused on the road surface, indicating he was closely watching. Moving slowly, the pedestrian was in front of the vehicle and appeared to be crossing. He wore a black T-shirt on his upper body and black slacks on his lower body. The weather was clear with bright brightness. The road surface was dry and leveled asphalt. The traffic volume on the two-way residential road was usual. Both sides of the road did not have a sidewalk or roadside strip, but there were street lights. This set of data provides a detailed description of the pedestrian and the surrounding environment in a third-person narrative style.",
            "caption_vehicle": "The vehicle is positioned on the right side of the pedestrian and is in close proximity to them. From the vehicle's field of view, the pedestrian is visible. The vehicle is going straight ahead at a speed of 5 km/h. The environment conditions include a male pedestrian in his 30s, standing at a height of 170 cm. He is wearing a black T-shirt for his upper body and black slacks for his lower body. The weather is clear and the brightness is bright. The road surface conditions are dry and the road is level with asphalt. The traffic volume is usual on this residential road with two-way traffic. There is no sidewalk available on both sides, and the roadside strip is also absent. Street lights are present in the surroundings.",
            "start_time": "33.641",
            "end_time": "37.678"
        },
        {
            "labels": [
                "2"
            ],
            "caption_pedestrian": "A man in his 30s wearing a black T-shirt and black slacks stands diagonally to the left of a vehicle on a residential road. His body is positioned perpendicular to the vehicle and to the right. The man's height is approximately 170 cm. The weather is clear, and the brightness is bright. The road surface is dry and level, made of asphalt. There are two-way traffic lanes, but there is no sidewalk or roadside strip on both sides of the road. Street lights illuminate the surroundings. Despite noticing the vehicle, the pedestrian, moving slowly, seems to be slowly looking around, possibly unaware of the vehicle's presence. The pedestrian is about to cross the road, intending to travel in front of the vehicle. The traffic volume is usual, creating a typical scenario for the pedestrian.",
            "caption_vehicle": "The vehicle is positioned diagonally to the right in front of the pedestrian and is at a close distance from them. The pedestrian is visible within the vehicle's field of view. The vehicle is currently stopped and its speed is 0 km/h. The driver is observing the surroundings and waiting for the appropriate moment to continue. Meanwhile, the environment conditions indicate that the pedestrian is a male in his 30s, approximately 170 cm tall, wearing a black T-shirt and black slacks. The weather is clear and the brightness is bright. The road conditions are favorable, with a dry asphalt surface and a level incline. The road is a residential road with two-way traffic and without sidewalks on both sides or roadside strips on both sides. However, there are street lights illuminating the area. The overall situation seems calm and ordinary, with the vehicle and pedestrian being momentarily paused in their respective positions on the road.",
            "start_time": "32.351",
            "end_time": "33.641"
        },
        {
            "labels": [
                "1"
            ],
            "caption_pedestrian": "The pedestrian, a male in his 30s, stood diagonally to the left in front of the vehicle. His body was perpendicular to the vehicle and to the right. He was approximately close to the vehicle, with a clear line of sight towards it. Slowly looking around, he was aware of the vehicle's presence. Dressed in a black T-shirt and slacks, he stood on a level and dry asphalt road. The weather was clear with bright lighting, and it was a usual day in terms of traffic volume. The event took place on a residential road with two-way traffic. There was no sidewalk on both sides, and neither the roadside strip nor the street lights were present. Despite this, the pedestrian was about to cross the road, moving slowly in the direction of travel in front of the vehicle.",
            "caption_vehicle": "The vehicle is positioned diagonally to the right in front of the pedestrian, at a close relative distance. The pedestrian is clearly visible within the vehicle's field of view. The vehicle is about to stop and is moving at a speed of 5 km/h. The driver is aware of the pedestrian's presence and is taking necessary action to halt the vehicle. The event takes place in a clear weather condition with bright brightness. The vehicle is traveling on a residential road with two-way traffic and a dry asphalt surface. The surrounding environment indicates a male pedestrian in his 30s, standing at a height of 170 cm. He is wearing a black T-shirt and black slacks. The road does not have a sidewalk on both sides and there is no roadside strip. However, street lights are present, ensuring sufficient visibility. The traffic volume is normal and the road is level. All these factors provide the necessary context for the event involving the vehicle and the pedestrian.",
            "start_time": "29.324",
            "end_time": "32.295"
        },
        {
            "labels": [
                "0"
            ],
            "caption_pedestrian": "The pedestrian, a male in his 30s, was standing diagonally to the left in front of the vehicle on a residential road. His body was oriented perpendicular to the vehicle and to the right. Positioned near the vehicle, his line of sight was directed towards his crossing destination. He was slowly looking around and dressed in a black T-shirt and slacks. The weather was clear and bright, with dry road surface conditions on the level asphalt road. Despite the usual traffic volume on the two-way street, there were no sidewalks or roadside strips available. However, street lights were present, ensuring visibility. The pedestrian's general action was going straight ahead, and his speed was slow. Overall, the pedestrian was actively observing his surroundings and preparing to cross the road, taking into account the environmental conditions and traffic situation.",
            "caption_vehicle": "The vehicle is positioned diagonally to the right in front of the pedestrian, at a close distance. The pedestrian is visible within the vehicle's field of view. The vehicle is going straight ahead at a speed of 10 km/h. Meanwhile, in the environment, there is a male pedestrian in his 30s, standing at a height of 170 cm. He is wearing a black T-shirt and black slacks. The weather is clear with bright lighting, and the road surface is dry and level. The road is a residential road with two-way traffic and does not have sidewalks on both sides. There are no roadside strips, but the street lights are on. Overall, the vehicle is in a normal traffic situation, with clear visibility of the pedestrian and suitable road conditions for its speed and direction.",
            "start_time": "28.258",
            "end_time": "29.324"
        }
    ]
}
```

- Test caption:

```jsx
{
  "id": 728,
  "event_phase": [
    {
      "labels": ["3"],
      "caption_pedestrian": "...",
      "caption_vehicle": "..."
    }
  ],
  "frame length": 86
}

```

- **Phase 0 Start Time From Video Beginning**

| **Metric** | **Value** |
| --- | --- |
| Min | 0.000 s |
| Max | 36.534 s |
| Mean | 19.410 s |
| Median | 23.280 s |
| P90 | 30.100 s |

Bảng này cho thấy phase 0 không phải lúc nào cũng bắt đầu tại giây 0 của video. Giá trị median là 23.28s, nghĩa là trong nhiều video, event thật sự chỉ bắt đầu sau một đoạn pre-event khá dài. Điều này khá quan trọng với overhead/fixed camera, vì video thường chứa cả bối cảnh trước khi pedestrian-vehicle interaction bắt đầu. Do đó, khi xử lý train/val, không nên giả định frame đầu tiên của video là frame đầu tiên của event. Cần dùng start_time và end_time trong caption JSON để cắt đúng đoạn phase.

- **Full Event Duration From Phase 0 to Phase 4**

| **Metric** | **Value** |
| --- | --- |
| Min | 2.030 s |
| Max | 32.440 s |
| Mean | 10.980 s |
| Median | 11.670 s |
| P90 | 17.010 s |

Một event đầy đủ từ phase 0 đến phase 4 thường kéo dài khoảng 11s, với mean 10.98s và median 11.67s. P90 là 17.01s, nghĩa là phần lớn event nằm trong khoảng dưới 17 giây, nhưng vẫn có một số event dài hơn 30 giây. Điều này cho thấy bài toán forecasting không chỉ là dự đoán vài frame ngắn hạn, mà có thể yêu cầu mô hình hiểu diễn tiến hành vi trong một khoảng thời gian tương đối dài. Về mặt training, cần có chiến lược sampling linh hoạt cho cả short event và long event.

- **Duration of Each Phase**

| **Phase** | **Min** | **Max** | **Mean** | **Median** | **P90** | **Main observation** |
| --- | --- | --- | --- | --- | --- | --- |
| Phase 0 | 0.167s | 6.782s | 1.274s | 1.019s | 2.103s | Short early phase |
| Phase 1 | 0.196s | 11.598s | 1.276s | 1.037s | 2.136s | Short, but has long outliers |
| Phase 2 | 0.100s | 5.644s | 1.164s | 1.070s | 1.887s | Shortest phase on average |
| Phase 3 | 0.200s | 11.546s | 3.532s | 3.087s | 7.733s | Long and highly variable action phase |
| Phase 4 | 0.200s | 9.525s | 3.711s | 3.723s | 5.550s | Long outcome/avoidance phase |

Bảng này cho thấy duration giữa các phase rất không đều. Phase 0, 1, 2 thường rất ngắn, median chỉ khoảng 1s, trong khi phase 3 và phase 4 dài hơn rõ rệt. Phase 3 có mean 3.53s và P90 7.73s, cho thấy giai đoạn action có độ biến thiên lớn. Phase 4 có mean 3.71s và median 3.72s, tức là outcome/avoidance thường kéo dài ổn định hơn phase đầu. 

- **Same Event, Different Timestamp Offsets Across Views**

| **View** | **Phase 0 start** | **Phase 4 end** | **Phase 0 to Phase 4 duration** |
| --- | --- | --- | --- |
| Overhead view | 28.258s | 42.275s | 14.017s |
| Vehicle view | 0.000s | 14.017s | 14.017s |

Bảng này minh họa một điểm dễ gây nhầm lẫn: cùng một event có thể có timestamp tuyệt đối khác nhau giữa overhead view và vehicle view. Trong ví dụ này, cả hai view đều mô tả event dài 14.017s, nhưng vehicle view bắt đầu từ 0s, còn overhead view bắt đầu phase 0 tại 28.258s. Nghĩa là vehicle video đã được trim để event bắt đầu ngay từ đầu, trong khi overhead video vẫn chứa đoạn trước event. 

- **Train/Validation Data Structure Summary**

| Dataset part | Split | Folder count | Overhead JSON | Vehicle JSON | Total JSON | Overhead videos | Vehicle videos | Total videos | Missing vehicle |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Main WTS | Train | 97 | 97 | 92 | 189 | 333 | 92 | 425 | 5 |
| Main WTS | Val | 48 | 48 | 47 | 95 | 163 | 47 | 210 | 1 |
| Main WTS | Train+Val | 145 | 145 | 139 | 284 | 496 | 139 | 635 | 6 |
| Normal Trimmed | Train | 70 | 70 | 44 | 114 | 70 | 44 | 114 | 26 |
| Normal Trimmed | Val | 34 | 34 | 27 | 61 | 34 | 27 | 61 | 7 |
| Normal Trimmed | Train+Val | 104 | 104 | 71 | 175 | 104 | 71 | 175 | 33 |
| BDD_PC_5K | Train | - | - | - | 2430 | - | - | 2430 | - |
| BDD_PC_5K | Val | - | - | 972 | 972 | - | 972 | 972 | - |
| BDD_PC_5K | Train+Val | - | - | 3402 | 3402 | - | 3402 | 3402 | - |

Trong **Main WTS**, tập train có 97 folders, còn tập validation có 48 folders, tổng cộng 145 folders. Phần này có 145 overhead caption JSON nhưng chỉ có 139 vehicle caption JSON, tức là có 6 folders thiếu vehicle view. Khi tính video files, Main WTS có 496 overhead videos và 139 vehicle videos, tổng cộng 635 videos.

Trong **Normal Trimmed**, tập train có 70 folders và validation có 34 folders, tổng cộng 104 folders. Tất cả 104 folders đều có overhead JSON, nhưng chỉ có 71 vehicle JSON, tức là có 33 folders thiếu vehicle view. Số video của Normal Trimmed khớp với số JSON: 104 overhead videos và 71 vehicle videos, tổng cộng 175 videos.

Riêng **BDD_PC_5K** là external dataset, không chia thành overhead/vehicle như WTS. BDD có 2430 videos/JSON ở train và 972 videos/JSON ở validation và tất cả đều là vehicles view, tổng cộng 3402 videos và 3402 caption JSON.

Kết luận chính là: **WTS Total có 810 videos và 459 caption JSON**, còn **BDD_PC_5K có 3402 videos/JSON**. Trong WTS, overhead view đầy đủ hơn vehicle view: có 249 overhead JSON nhưng chỉ 210 vehicle JSON, cho thấy không phải folder WTS nào cũng có vehicle view. Normal Trimmed là một phần đáng kể của WTS, đóng góp 104/249 folders và 175/810 videos, nên không nên bỏ qua khi training hoặc phân tích dataset.

- **Caption Statistics Across Train/Validation Sets**

| Dataset part | JSON files | Entries / segments | Entries per JSON | Combined words mean | Combined words median | Combined words P90 | Pedestrian words mean | Vehicle words mean | Label type |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BDD_PC_5K external | 3402 | 17011 | 3401 files x 5, 1 file x 6 | 277.72 | 279 | 306 | 141.12 | 136.60 | textual phase labels |
| WTS main | 284 | 1420 | 284 files x 5 | 268.78 | 278 | 315 | 139.05 | 129.73 | numeric labels 0-4 |
| WTS normal_trimmed | 175 | 875 | 175 files x 5 | 274.92 | 276 | 303 | 143.25 | 131.67 | numeric labels 0-4 |
| Total | 3861 | 19306 | mostly 5 entries/file | - | - | - | - | - | mixed |

Trong **BDD_PC_5K external**, có 3402 file caption JSON và tổng cộng 17011 caption entries/segments. Hầu hết các file có đúng 5 entries, cụ thể là 3401 file có 5 entries và chỉ có 1 file có 6 entries. Caption của BDD khá dài: trung bình combined caption có 277.72 words, median 279 words, và P90 306 words. Khi tách riêng, pedestrian caption có trung bình 141.12 words, còn vehicle caption có trung bình 136.60 words. Label của BDD dùng dạng text phase labels, ví dụ prerecognition, recognition, judgement, action, avoidance.

Trong **WTS main**, có 284 caption JSON và 1420 entries/segments. Mỗi JSON có đúng 5 entries, nên 284 x 5 = 1420. Caption trong WTS main cũng dài, với combined caption trung bình 268.78 words, median 278 words, và P90 315 words. Pedestrian caption trung bình 139.05 words, vehicle caption trung bình 129.73 words. Label của WTS main là numeric labels từ 0 đến 4.

Trong **WTS normal_trimmed**, có 175 caption JSON và 875 entries/segments, cũng theo cấu trúc 175 x 5. Caption của normal_trimmed có độ dài trung bình 274.92 words, median 276 words, và P90 303 words. Pedestrian caption trung bình 143.25 words, vehicle caption trung bình 131.67 words. Giống WTS main, phần này dùng numeric labels 0-4.

Tổng cộng toàn bộ caption có 3861 caption JSON và 19306 caption entries/segments. Phần lớn dữ liệu đến từ **BDD_PC_5K**, với 3402/3861 JSON files. Tuy nhiên, độ dài caption giữa BDD và WTS khá tương đồng: combined captions đều nằm khoảng 269-278 words trung bình. 

- **Phase Label Distribution in Train/Validation Captions**

| Dataset part | Label scheme | Label | Count |
| --- | --- | --- | --- |
| WTS main | numeric | 0 | 284 |
| WTS main | numeric | 1 | 284 |
| WTS main | numeric | 2 | 284 |
| WTS main | numeric | 3 | 284 |
| WTS main | numeric | 4 | 284 |
| WTS normal_trimmed | numeric | 0 | 175 |
| WTS normal_trimmed | numeric | 1 | 175 |
| WTS normal_trimmed | numeric | 2 | 175 |
| WTS normal_trimmed | numeric | 3 | 175 |
| WTS normal_trimmed | numeric | 4 | 175 |
| BDD_PC_5K | textual | prerecognition | 3402 |
| BDD_PC_5K | textual | recognition | 3402 |
| BDD_PC_5K | textual | judgement | 3403 |
| BDD_PC_5K | textual | action | 3402 |
| BDD_PC_5K | textual | avoidance | 3402 |

Trong **WTS main**, label được biểu diễn bằng số từ 0 đến 4. Mỗi label đều xuất hiện đúng 284 lần. Điều này cho thấy WTS main có phân phối phase hoàn toàn cân bằng: mỗi caption JSON có đủ 5 phase labels, và tổng cộng có 284 JSON nên mỗi label có 284 entries.

Trong **WTS normal_trimmed**, label cũng là numeric labels từ 0 đến 4. Mỗi label xuất hiện đúng 175 lần. Tương tự WTS main, điều này nghĩa là mỗi normal_trimmed JSON cũng có đủ 5 phase labels, và phân phối phase cân bằng tuyệt đối.

Trong **BDD_PC_5K**, label không dùng số mà dùng textual phase labels: prerecognition, recognition, judgement, action, và avoidance. Hầu hết mỗi label xuất hiện 3402 lần, tương ứng với 3402 JSON files. Riêng label judgement xuất hiện 3403 lần vì có một file BDD chứa 6 entries thay vì 5.

Kết luận chính là: cả WTS và BDD đều có cấu trúc phase rất rõ ràng và gần như cân bằng. WTS dùng numeric labels 0-4, còn BDD dùng textual labels tương ứng với các giai đoạn hành vi.

- ***Phase-to-Phase Caption Overlap in Train/Validation Sets***

| Dataset part | Split | View | Pairwise overlap mean | Median | Min | Max | P90 | Interpretation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BDD_PC_5K | Train | bdd | 0.490 | 0.488 | 0.218 | 0.776 | 0.579 | High static context reuse |
| BDD_PC_5K | Val | bdd | 0.492 | 0.492 | 0.250 | 0.771 | 0.583 | High static context reuse |
| WTS main | Train | overhead | 0.465 | 0.463 | 0.252 | 0.827 | 0.562 | Moderate phase change |
| WTS main | Train | vehicle | 0.466 | 0.465 | 0.252 | 0.827 | 0.562 | Similar to overhead captions |
| WTS main | Val | overhead | 0.461 | 0.455 | 0.190 | 0.704 | 0.566 | Moderate phase change |
| WTS main | Val | vehicle | 0.461 | 0.457 | 0.190 | 0.704 | 0.566 | Similar to overhead captions |
| WTS normal_trimmed | Train | overhead | 0.505 | 0.505 | 0.303 | 0.733 | 0.589 | More repeated/static than WTS main |
| WTS normal_trimmed | Train | vehicle | 0.504 | 0.508 | 0.303 | 0.733 | 0.588 | More repeated/static than WTS main |
| WTS normal_trimmed | Val | overhead | 0.509 | 0.508 | 0.326 | 0.704 | 0.590 | Highest overlap |
| WTS normal_trimmed | Val | vehicle | 0.504 | 0.500 | 0.326 | 0.663 | 0.583 | High overlap |

Chỉ số overlap ở đây nằm từ 0 đến 1:

- 0 nghĩa là gần như không trùng nội dung.
- 1 nghĩa là gần như trùng hoàn toàn.
- Khoảng 0.46-0.51 nghĩa là gần một nửa content tokens được lặp lại giữa các phase.

Bảng này đo mức độ giống nhau giữa các caption của **các phase khác nhau trong cùng một event**. Với mỗi phase, ghép hai caption lại:

`phase caption = caption_pedestrian + caption_vehicle`

Sau đó so sánh caption của phase 0, 1, 2, 3, 4 với nhau. Vì vậy, overlap ở đây cho biết: trong cùng một event, các phase caption lặp lại bao nhiêu nội dung.

Trong **BDD_PC_5K**, train có mean overlap 0.490, validation có mean overlap 0.492. Hai giá trị này gần như giống nhau, cho thấy cấu trúc caption của BDD train và val khá ổn định. Overlap khoảng 0.49 nghĩa là các phase trong cùng một event chia sẻ gần một nửa lượng content words. Phần được lặp lại thường là thông tin static như điều kiện đường, thời tiết, quần áo người đi bộ, vị trí tương đối, môi trường giao thông. Phần còn lại thay đổi theo phase, ví dụ người đi bộ nhận ra xe, bắt đầu di chuyển, xe giảm tốc, hoặc hành vi tránh va chạm.

Trong **WTS main**, overlap thấp hơn: train overhead 0.465, train vehicle 0.466, val overhead 0.461, val vehicle 0.461. Điều này cho thấy caption giữa các phase trong WTS main thay đổi mạnh hơn BDD. Nói cách khác, phase 0 và phase 4 trong WTS main khác nhau rõ hơn về mặt hành vi. Đây là hợp lý vì WTS main chứa nhiều tình huống staged conflict hoặc accident-like, nên từ giai đoạn đầu đến giai đoạn cuối, trạng thái pedestrian/vehicle thay đổi nhiều hơn.

Trong **WTS normal_trimmed**, overlap cao nhất: train overhead 0.505, train vehicle 0.504, val overhead 0.509, val vehicle 0.504. Điều này cho thấy các phase caption trong normal_trimmed lặp lại nhiều nội dung hơn. Có thể hiểu là normal_trimmed ổn định hơn WTS main: bối cảnh và hành vi giữa các phase ít thay đổi mạnh, nên caption của các phase giống nhau hơn.

Một điểm đáng chú ý là trong WTS main, overlap của overhead view và vehicle view gần như bằng nhau. Ví dụ train overhead 0.465 và train vehicle 0.466; val overhead 0.461 và val vehicle 0.461. Điều này cho thấy annotation text giữa hai view có cấu trúc khá nhất quán. Dù video view khác nhau, caption vẫn mô tả event theo cùng logic phase.

Tóm lại:

| **Dataset part** | **Mean overlap** | **Ý nghĩa** |
| --- | --- | --- |
| WTS normal_trimmed | ~0.50 | Phase captions giống nhau nhiều nhất, hành vi ổn định hơn |
| BDD_PC_5K | ~0.49 | Caption structure ổn định, nhiều static context lặp lại |
| WTS main | ~0.46 | Phase captions thay đổi nhiều hơn, event dynamics mạnh hơn |

Kết luận quan trọng cho method là caption không nên được xem như một đoạn text tự do hoàn toàn độc lập giữa các phase. Trong mỗi caption có hai lớp thông tin:

| **Loại thông tin** | **Ví dụ** | **Vai trò** |
| --- | --- | --- |
| Static context | quần áo, đường, thời tiết, vị trí tương đối, traffic volume | Giữ bối cảnh nhất quán |
| Dynamic behavior | awareness, crossing, stopping, speed change, collision, avoidance | Quyết định chuyển động tương lai |

- ***Caption Overlap by Phase Pair in WTS Main***

| Label pair | Mean overlap | Median | Min | Max | P90 | Interpretation |
| --- | --- | --- | --- | --- | --- | --- |
| 0-1 | 0.524 | 0.527 | 0.237 | 0.827 | 0.615 | Most similar adjacent early phases |
| 0-2 | 0.490 | 0.492 | 0.331 | 0.683 | 0.561 | Still similar, shared context |
| 0-3 | 0.469 | 0.468 | 0.252 | 0.657 | 0.555 | More behavior changes |
| 0-4 | 0.415 | 0.416 | 0.271 | 0.573 | 0.483 | Largest semantic shift |
| 1-2 | 0.498 | 0.508 | 0.190 | 0.655 | 0.588 | Adjacent phase similarity |
| 1-3 | 0.479 | 0.478 | 0.230 | 0.676 | 0.573 | Moderate shift |
| 1-4 | 0.414 | 0.414 | 0.247 | 0.573 | 0.487 | Strong shift to outcome |
| 2-3 | 0.496 | 0.500 | 0.265 | 0.662 | 0.578 | Adjacent mid/action phases |
| 2-4 | 0.420 | 0.421 | 0.268 | 0.589 | 0.500 | Strong shift |
| 3-4 | 0.431 | 0.431 | 0.265 | 0.587 | 0.509 | Action to outcome shift |

Kết quả cho thấy có một xu hướng rất rõ: **các phase gần nhau thì caption giống nhau hơn, còn phase cách xa nhau thì caption khác nhau hơn**.

Cặp có overlap cao nhất là **label 0-1**, với mean 0.524 và median 0.527. Điều này hợp lý vì label 0 và label 1 đều nằm ở giai đoạn đầu event. Cả hai thường vẫn mô tả cùng bối cảnh tĩnh như pedestrian appearance, road condition, vehicle position, weather, và traffic environment. Hành vi có thay đổi, nhưng chưa quá mạnh.

Các cặp gần nhau khác như **1-2** và **2-3** cũng có overlap tương đối cao:

- label 1-2 mean 0.498
- label 2-3 mean 0.496

Điều này cho thấy progression giữa các phase là liên tục, không phải các caption hoàn toàn rời rạc. Từ phase này sang phase tiếp theo, nhiều thông tin static vẫn được giữ lại, chỉ phần hành vi thay đổi dần.

Ngược lại, các cặp có label 4 thường overlap thấp hơn nhiều:

- label 0-4 mean 0.415
- label 1-4 mean 0.414
- label 2-4 mean 0.420
- label 3-4 mean 0.431

Điều này rất quan trọng vì label 4 là phase cuối/outcome, thường chứa những mô tả như collision, avoidance, hoặc kết quả hành động. Do đó caption label 4 khác rõ hơn so với các phase đầu.

Cặp **0-4** và **1-4** là hai cặp có mean overlap thấp nhất, khoảng 0.414-0.415. Đây là bằng chứng rằng trong WTS main, event thực sự có thay đổi semantic đáng kể từ phase đầu đến phase cuối. Nói cách khác, label không chỉ là metadata hình thức; nó phản ánh tiến trình hành vi trong event.

Một điểm nữa là **P90 của label 0-1 là 0.615**, khá cao. Điều này nghĩa là ở 10% trường hợp, label 0 và 1 cực kỳ giống nhau về nội dung. Trong khi đó P90 của 0-4 chỉ 0.483, thấp hơn rõ rệt. Điều này củng cố kết luận rằng phase đầu gần nhau có nhiều static/context reuse, còn phase cuối khác biệt hơn.

Kết luận chính:

`Overlap cao nhất: 0-1 = 0.524
Overlap thấp nhất: 1-4 = 0.414
Các cặp với label 4 thường overlap thấp`

Ý nghĩa cho method:

- Model nên học **temporal phase progression**, không xem các caption phase như độc lập.
- Label/phase embedding là có cơ sở, vì caption thay đổi có quy luật từ 0 đến 4.
- Với forecasting, caption của phase sau chứa nhiều dynamic/outcome information hơn, đặc biệt label 4.
- Khi training, có thể tạo transition pairs như phase 0 -> phase 1, phase 1 -> phase 2, nhưng cần chú ý rằng chuyển sang phase 4 là semantic shift mạnh nhất.

- **Table 1: Test Set Overview**

| Metric | Value | Meaning |
| --- | --- | --- |
| Test samples | 71 | Số folder test độc lập |
| Total history frames K | 3584 | Tổng số input frames trong toàn test set |
| Total target frames N | 5194 | Tổng số frames cần generate |
| Input resolution | 1280x720 | Tất cả 71 samples cùng resolution |
| Warnings | 0 | Không thiếu input/caption/frame length |
| Event phase entries per caption | 1 | Mỗi test caption chỉ có 1 event_phase |

Test set có tổng cộng 71 samples, nghĩa là có 71 folder test độc lập. Mỗi folder là một bài dự đoán riêng, gồm input/ chứa history frames và một file caption.json mô tả future scene cần generate. Các folder này không cần liên kết với nhau khi inference.

Tổng số **history frames K** là 3584. Đây là tổng số input frames có sẵn trong toàn bộ test set. Nói cách khác, nếu cộng tất cả ảnh trong các folder input/, ta được 3584 frames. Số history frames ở mỗi sample có thể khác nhau, nhưng tổng toàn test là 3584.

Tổng số **target frames N** là 5194. Đây là tổng số frames mà model cần generate để nộp bài. Với mỗi test sample, số frame cần sinh được lấy từ key frame length trong caption.json. Khi cộng frame length của toàn bộ 71 samples, ta được 5194.

Tất cả input frames trong test set đều có resolution 1280x720. Điều này rất quan trọng vì submission yêu cầu generated frames cũng phải có cùng resolution. Vì vậy output của model cho mọi sample phải là ảnh PNG 1280x720.

Cuối cùng, mỗi caption.json trong test chỉ có đúng 1 event_phase. Điều này khác với train/val, nơi mỗi JSON thường có 5 phase labels. Ở test, mỗi sample chỉ yêu cầu generate future frames cho một phase cụ thể đã được cung cấp. Vì vậy model chỉ cần đọc đúng caption trong phase đó và generate đúng N frames tương ứng.

Tóm lại, test set là **sample-centric**: mỗi folder là một test case độc lập, có history frames riêng, caption riêng, và target length riêng. Inference pipeline phải đọc từng folder, lấy số input frames K, lấy frame length = N, rồi output đúng N frames trong folder prediction tương ứng.

- **Table 2: Test Sample Type Distribution**

| Sample type | Samples | Sample ratio | Total target frames | Frame ratio |
| --- | --- | --- | --- | --- |
| BDD_PC_5K | 44 | 61.97% | 2892 | 55.68% |
| WTS vehicle/IP camera | 20 | 28.17% | 1774 | 34.15% |
| WTS overhead/fixed camera | 7 | 9.86% | 528 | 10.17% |
| Total | 71 | 100% | 5194 | 100% |

Bảng này cho thấy test set không chỉ gồm một loại dữ liệu duy nhất, mà là một tập **mixed-domain** gồm BDD_PC_5K, WTS vehicle/IP camera và WTS overhead/fixed camera.

Nhóm lớn nhất là **BDD_PC_5K**, với 44 samples trên tổng 71 samples, chiếm 61.97% test set. Nhóm này cũng chiếm 2892 target frames trên tổng 5194 frames cần generate, tương đương 55.68%. Điều này cho thấy BDD không phải là phần phụ nhỏ trong test, mà là nhóm chiếm đa số cả về số lượng sample lẫn số lượng frame cần sinh.

Nhóm thứ hai là **WTS vehicle/IP camera**, gồm 20 samples, chiếm 28.17% số sample. Tuy nhiên nhóm này cần generate 1774 frames, chiếm 34.15% tổng số target frames. Tỷ lệ frame cao hơn tỷ lệ sample cho thấy các sample vehicle/IP thường có target horizon dài hơn trung bình. Đây có thể là nhóm khó vì camera có thể có ego-motion hoặc chuyển động mạnh hơn overhead view.

Nhóm nhỏ nhất là **WTS overhead/fixed camera**, chỉ có 7 samples, chiếm 9.86% test set. Nhóm này cần generate 528 frames, tương đương 10.17% tổng target frames. Mặc dù overhead view quan trọng trong WTS training, ở test nó chỉ chiếm một phần nhỏ.

Tổng cộng, test set có 71 samples và 5194 target frames cần generate. Phân phối này cho thấy leaderboard không chỉ phụ thuộc vào khả năng xử lý WTS overhead, mà phụ thuộc nhiều hơn vào khả năng xử lý **BDD/front-view** và **WTS vehicle/IP camera**.

Kết luận quan trọng cho method là: BDD_PC_5K cần được xem là một domain chính trong training hoặc fine-tuning, không chỉ là pretraining phụ. Đồng thời, pipeline inference cần robust với vehicle/front-view samples vì nhóm WTS vehicle/IP chiếm hơn một phần ba tổng số frames cần generate.

- **Table 3: K / N / Caption Global Stats**

| Metric | Count | Min | Max | Mean | Median | P90 | Sum |
| --- | --- | --- | --- | --- | --- | --- | --- |
| History frames K | 71 | 10 | 224 | 50.48 | 40 | 101 | 3584 |
| Target frames N | 71 | 51 | 120 | 73.15 | 67 | 106 | 5194 |
| Total caption words | 71 | 140 | 322 | 268.62 | 274 | 300 | 19072 |
| Pedestrian caption words | 71 | 72 | 168 | 137.14 | 140 | 151 | 9737 |
| Vehicle caption words | 71 | 44 | 170 | 131.48 | 136 | 153 | 9335 |

Bảng này mô tả phân phối của ba nhóm thông tin quan trọng trong test set: số **history frames K**, số **target frames N**, và độ dài caption.

Đầu tiên, **History frames K** là số input frames có trong folder input/ của mỗi test sample. Có tổng cộng 71 samples. Sample ít nhất có 10 history frames, nhiều nhất có 224 history frames. Trung bình mỗi sample có 50.48 history frames, median là 40, và P90 là 101. Điều này nghĩa là 90% test samples có tối đa 101 history frames, nhưng vẫn có một số sample rất dài. Tổng số history frames trên toàn bộ test set là 3584.

Tiếp theo, **Target frames N** là số frames mà model cần generate, lấy từ frame length trong caption.json. N nhỏ nhất là 51, lớn nhất là 120. Trung bình mỗi sample cần generate 73.15 frames, median là 67, và P90 là 106. Tổng cộng toàn test set cần generate 5194 frames. Điều này cho thấy test không có fixed prediction length; mỗi sample yêu cầu một số future frames khác nhau.

Phần caption cũng rất đáng chú ý. **Total caption words**, tức tổng số words của pedestrian caption và vehicle caption, có min 140, max 322, mean 268.62, median 274, và P90 300. Như vậy caption trong test khá dài và giàu thông tin. Tổng cộng toàn test set có 19072 words.

Khi tách riêng, **pedestrian caption** có trung bình 137.14 words, median 140, và P90 151. **Vehicle caption** có trung bình 131.48 words, median 136, và P90 153. Hai phần caption có độ dài tương đối cân bằng, cho thấy cả pedestrian behavior lẫn vehicle behavior đều chứa nhiều thông tin quan trọng.

Kết luận chính từ bảng này là test set có tính **variable-length** rất rõ rệt:

- K thay đổi từ 10 đến 224;
- N thay đổi từ 51 đến 120;
- caption dài khoảng 269 words trung bình.

Vì vậy inference pipeline không nên hard-code số history frames hoặc số target frames. Model cần hỗ trợ variable-K conditioning, variable-N generation, và text encoder/prompting strategy phải xử lý caption dài một cách ổn định.

- **Table 4: Stats By Sample Type**

| Sample type | Samples | K min | K max | K mean | K median | K P90 | N min | N max | N mean | N median | N P90 | Target frames | Caption mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BDD_PC_5K | 44 | 10 | 85 | 35.05 | 32 | 59 | 51 | 90 | 65.73 | 62.5 | 84 | 2892 | 276.07 |
| WTS overhead/fixed camera | 7 | 42 | 155 | 90.57 | 55 | 155 | 52 | 118 | 75.43 | 66 | 118 | 528 | 261.43 |
| WTS vehicle/IP camera | 20 | 21 | 224 | 70.40 | 51 | 120 | 52 | 120 | 88.70 | 87 | 119 | 1774 | 254.75 |

Bảng này phân tích chi tiết test set theo từng loại sample: **BDD_PC_5K**, **WTS overhead/fixed camera**, và **WTS vehicle/IP camera**. Nó cho thấy mỗi nhóm có đặc điểm rất khác nhau về số history frames K, số target frames N, và độ dài caption.

Với **BDD_PC_5K**, có 44 samples, là nhóm lớn nhất trong test set. Số history frames K dao động từ 10 đến 85, trung bình 35.05, median 32, và P90 59. Số target frames N dao động từ 51 đến 90, trung bình 65.73, median 62.5, và P90 84. Tổng cộng nhóm BDD cần generate 2892 frames. Caption của BDD có độ dài trung bình 276.07 words, cao nhất trong ba nhóm. Điều này cho thấy BDD samples có history và target horizon vừa phải, nhưng caption dài và chiếm phần lớn số lượng test samples.

Với **WTS overhead/fixed camera**, chỉ có 7 samples, là nhóm nhỏ nhất. Tuy nhiên số history frames khá lớn: K nhỏ nhất là 42, lớn nhất 155, mean 90.57, median 55, và P90 155. Số target frames N dao động từ 52 đến 118, mean 75.43, median 66, và P90 118. Tổng số target frames của nhóm này là 528, caption mean là 261.43 words. Dù overhead samples ít, chúng có thể cung cấp history dài, nên cần frame selection nếu model không thể dùng toàn bộ input frames.

Với **WTS vehicle/IP camera**, có 20 samples. Đây là nhóm khó nhất về temporal length. History frames K dao động từ 21 đến 224, mean 70.40, median 51, và P90 120. Target frames N dao động từ 52 đến 120, mean 88.70, median 87, và P90 119. Tổng cộng nhóm này cần generate 1774 frames. Caption mean là 254.75 words, thấp hơn BDD một chút nhưng vẫn rất dài. Nhóm này có target horizon dài nhất và history max cao nhất, nên rất quan trọng cho inference strategy.

So sánh ba nhóm:

- **BDD_PC_5K** chiếm nhiều sample nhất và nhiều target frames nhất tổng thể, nhưng K/N trung bình thấp hơn WTS vehicle/IP.
- **WTS overhead/fixed camera** có ít sample nhất nhưng history frames có thể rất dài.
- **WTS vehicle/IP camera** có N trung bình cao nhất (88.70) và max K cao nhất (224), nên là nhóm rủi ro nhất về long-horizon generation và camera motion.

- **Table 5: Test Label Counts**

| Label | Count | Ratio | Interpretation |
| --- | --- | --- | --- |
| 1 | 14 | 19.72% | Early/mid event phase |
| 2 | 4 | 5.63% | Mid event phase |
| 3 | 27 | 38.03% | Action-heavy phase |
| 4 | 26 | 36.62% | Late/outcome phase |
| 0 | 0 | 0.00% | Not present in test |
| Total | 71 | 100% | All test samples |

Trong test set, mỗi caption.json chỉ có một event_phase, và mỗi event_phase có một label. Tổng cộng có 71 test samples, nên tổng số label cũng là 71.

Label xuất hiện nhiều nhất là **label 3**, với 27 samples, chiếm 38.03% test set. Label 3 thường tương ứng với giai đoạn hành động rõ ràng hơn trong event, ví dụ pedestrian đang crossing, vehicle đang di chuyển/chậm lại, hoặc hai đối tượng đang tương tác trực tiếp.

Label xuất hiện nhiều thứ hai là **label 4**, với 26 samples, chiếm 36.62%. Label 4 thường là giai đoạn cuối hoặc outcome của event, ví dụ avoidance, collision, hoặc trạng thái sau hành động chính. Nếu gộp label 3 và label 4 lại, ta có:

`27 + 26 = 53 samples
53 / 71 ≈ 74.65%`

Tức là gần **75% test set nằm ở các phase muộn/action/outcome**.

Label 1 có 14 samples, chiếm 19.72%. Đây có thể được hiểu là early/mid event phase, nơi pedestrian hoặc vehicle bắt đầu có dấu hiệu nhận biết/tương tác.

Label 2 chỉ có 4 samples, chiếm 5.63%, là nhóm rất nhỏ trong test set.

Đặc biệt, **label 0 không xuất hiện trong test**. Điều này nghĩa là test set không đánh giá phase đầu tiên/pre-recognition. Model chủ yếu bị kiểm tra ở các đoạn đã có hành động hoặc outcome rõ hơn.

Kết luận chính là phân phối label trong test **không cân bằng** như train/val. Train/val có đủ label 0-4 cân bằng, nhưng test tập trung rất mạnh vào label 3 và 4. 

- **Table 6: Top Samples By History K**

| Rank | Sample | Type | K | N | Label |
| --- | --- | --- | --- | --- | --- |
| 1 | 20231006_18_CN29_T1_192.168.0.11_1 | WTS vehicle/IP | 224 | 119 | 4 |
| 2 | 20231013_101824_normal_192.168.0.13_4_event_0 | WTS vehicle/IP | 175 | 100 | 4 |
| 3 | 20230728_8_CN29_T1_Camera4_2 | WTS overhead | 155 | 87 | 4 |
| 4 | 20230728_13_CN21_T1_Camera1_0 | WTS overhead | 146 | 66 | 4 |
| 5 | 20230728_13_CN21_T1_Camera1_4 | WTS overhead | 146 | 66 | 4 |
| 6 | 20230929_70_SN34_T1_192.168.0.12_4 | WTS vehicle/IP | 120 | 72 | 4 |
| 7 | 20231013_101845_normal_192.168.0.12_4_event_1 | WTS vehicle/IP | 101 | 101 | 1 |
| 8 | 20231013_101845_normal_192.168.0.12_4_event_1_2 | WTS vehicle/IP | 101 | 84 | 3 |
| 9 | 20231013_112853_normal_192.168.0.11_2_event_0_2 | WTS vehicle/IP | 89 | 85 | 4 |
| 10 | video2189 | BDD_PC_5K | 85 | 87 | 3 |

**Table 7: Top Samples By Target N**

| Rank | Sample | Type | K | N | Label |
| --- | --- | --- | --- | --- | --- |
| 1 | 20230929_70_SN34_T1_192.168.0.12_3 | WTS vehicle/IP | 31 | 120 | 3 |
| 2 | 20230929_70_SN34_T1_192.168.0.13_4 | WTS vehicle/IP | 31 | 120 | 3 |
| 3 | 20231006_18_CN29_T1_192.168.0.11_1 | WTS vehicle/IP | 224 | 119 | 4 |
| 4 | 20230707_16_CN10_T1_Camera1_3 | WTS overhead | 47 | 118 | 3 |
| 5 | 20231013_104036_normal_192.168.0.11_4_event_1 | WTS vehicle/IP | 70 | 115 | 4 |
| 6 | 20230929_66_SN2_T1_192.168.0.11_1 | WTS vehicle/IP | 32 | 106 | 3 |
| 7 | 20230929_66_SN2_T1_192.168.0.12_2 | WTS vehicle/IP | 32 | 106 | 3 |
| 8 | 20230929_67_SN3_T1_192.168.0.12_2 | WTS vehicle/IP | 31 | 106 | 4 |
| 9 | 20231013_101845_normal_192.168.0.12_4_event_1 | WTS vehicle/IP | 101 | 101 | 1 |
| 10 | 20231013_101824_normal_192.168.0.13_4_event_0 | WTS vehicle/IP | 175 | 100 | 4 |
- **Table 8: Caption Length By Sample Type**

| **Sample type** | **Samples** | **Combined words mean** | **Combined words median** | **Combined words P90** | **Pedestrian words mean** | **Vehicle words mean** | **Nhận xét** |
| --- | --- | --- | --- | --- | --- | --- | --- |
| All test | 71 | 274.45 | 280.00 | 309.00 | 139.77 | 134.68 | Hai caption khá cân bằng |
| BDD_PC_5K | 44 | 281.84 | 284.00 | 304.70 | 142.91 | 138.93 | Caption dài nhất trung bình |
| WTS overhead/fixed | 7 | 266.29 | 275.00 | 307.00 | 137.86 | 128.43 | Ngắn hơn BDD |
| WTS vehicle/IP | 20 | 261.05 | 266.50 | 309.00 | 133.55 | 127.50 | Ngắn nhất trung bình |
- **Table 9: Pedestrian Vehicle Caption Overlap**

| **Sample type** | **Samples** | **Overlap mean** | **Overlap median** | **Overlap min** | **Overlap max** | **Overlap P90** | **Interpretation** |
| --- | --- | --- | --- | --- | --- | --- | --- |
| All test | 71 | 0.358 | 0.367 | 0.053 | 0.512 | 0.451 | Có lặp static context, nhưng không trùng hoàn toàn |
| BDD_PC_5K | 44 | 0.379 | 0.383 | 0.091 | 0.512 | 0.473 | Lặp nhiều nhất giữa pedestrian và vehicle caption |
| WTS overhead/fixed | 7 | 0.292 | 0.293 | 0.174 | 0.398 | 0.387 | Hai caption khác nhau nhiều hơn |
| WTS vehicle/IP | 20 | 0.334 | 0.343 | 0.053 | 0.424 | 0.408 | Mức overlap trung bình, có vài case rất lệch |

Caption của test set khá dài, trung bình 274.45 words/sample. Pedestrian caption trung bình 139.77 words, vehicle caption trung bình 134.68 words, tức là hai nguồn text có độ dài gần ngang nhau. BDD_PC_5K có caption dài nhất trung bình 281.84 words, trong khi WTS overhead/fixed là 266.29 và WTS vehicle/IP là 261.05.

Overlap giữa pedestrian và vehicle caption trung bình là 0.358. Điều này cho thấy hai caption có lặp lại static context như road, weather, pedestrian appearance, road condition, traffic condition, nhưng không phải bản sao của nhau. BDD có overlap cao nhất 0.379, nghĩa là phần text giữa hai caption lặp lại nhiều hơn. WTS overhead/fixed thấp nhất 0.292, tức là pedestrian caption và vehicle caption bổ sung thông tin khác nhau nhiều hơn.
# twQTeamImageProcessor

## Usage
0. Import
    ```python
    from track_detection.twQTeamImageProcessor import twQTeamImageProcessor
    ```
1. Init
    ```python
    def __init__(self, car, record_folder = None):
        ...
        self.m_twQTeamImageProcessor = twQTeamImageProcessor(track_mode=0)
    ```
2. Exec
    ```python
    def on_dashboard(self, src_img, last_steering_angle, speed, throttle, info):
        ...
        self.m_twQTeamImageProcessor.processImage(src_img)
        bIsReverseTrack   = self.m_twQTeamImageProcessor.isReverseTrack()
        current_track_idx = self.m_twQTeamImageProcessor.currentTrackIndex()

        if bIsReverseTrack == True:
            print('Reverse: ' + str(self.m_twQTeamImageProcessor.m_cnt_reverse))
        if current_track_idx != -1:
            self.m_twQTeamImageProcessor.showCurrentTrack(current_track_idx)
        else:
            print('Cannot find current track')
    ```
# WebQore demo

## 使い方

### WebQoreのAPIアカウント申請とQoreSDKのダウンロード

APIの無料アカウント（2ヶ月間）を発行しますので、以下のURLから「利用タイプ」で「Qiitaアドベントカレンダ執筆」を選択し、Qiitaアカウント名、メールアドレスを入力し送信ください。

https://docs.google.com/forms/d/e/1FAIpQLSffXV4AQQFsj_SsGujBvYgCt3qhEb4Id-Gv3QvWOV5iVp1meA/viewform

OS・PythonのバージョンにあったQoreSDKをダウンロードし、プロジェクトの直下に配置する。

例）Ubuntu18.04・Python3.6

qore_sdk-0.1.0-cp36-cp36m-linux_x86_64.whl 

### 依存ライブラリのインストール

- Using Pipenv

```
pipenv sync
```

- Using pip

```
pip install -r requirements.txt
```
## neo4j-playlist-builder
A tool to generate spotify playlists using Neo4j graph data science.
Taking a single (large, public) playlist as an input, the tool uses node similarity algorithms and kmeans clustering to group songs based on genre, energy and mood.


### Quick setup
This project uses a single Python script: `neo4j_spotify_playlist_builder.py`.
In here, you specify the parameters needed to connect to your spotify account using the Spotify API:
```
user_id = "[ADD YOUR SPOTIFY USER ID HERE]"
client = "[ADD YOUR SPOTIFY CLIENT ID HERE]"
secret = "[ADD YOUR SPOTIFY CLIENT SECRET HERE]"
playlist_uri = "[ADD YOUR PUBLIC PLAYLIST TO SORT HERE]"
```

Steps:
1. Your `user_id` can be found by using the web version of spotify and going to your profile overview. A spotify user id will look something like this: `111306XXXXX`
2. You need the Spotify developer dashboard to obtain a client id/secret. You can access it here: https://developer.spotify.com/dashboard/login. Next, create an app to obtain a `client_id` and a `client_secret`.
3. Your public `playlist_uri` can be found using the spotify application. Right click a playlist, select 'Share' and click 'Copy Spotify URI'. Your URI will have the following format: `spotify:playlist:XXXXXXXXXXXXXXXXXX`
4. Ensure that your Spotify developer app has the right `redirect_url` configured. For this tool to work, go to the Spotify developer dashboard and open the app you created. Click 'edit settings', and add the following url to "Redirect URIs": http://localhost:8888/callback.
5. Set up your Neo4j connection in `neo4j_spotify_playlist_builder.py`:
    ```
    neo4j_url = "bolt://localhost:7687"
    neo4j_username = "neo4j"
    neo4j_password = "neo" 
    ```
    Keep in mind this application clears your database, so best use a fresh DB.
6. Install python dependencies in `requirements.txt`.
7. Run `neo4j_spotify_playlist_builder.py` and watch the magic happen!

### Notes
An input playlist *must* be made public for the tool to work. Output playlists will be private by default. 
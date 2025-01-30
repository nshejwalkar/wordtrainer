document.getElementById('fileInput').addEventListener('change', async function() {
   // event.preventDefault();  // prevents auto reload on ok click (this is not it, it was liveserver auto reloading that was the issue)

   const file = this.files[0];
   const fileNameElement = document.getElementById('fileName')
   const errorElement = document.getElementById('errorMessage')
   const successElement = document.getElementById('successMessage')
   const boardElement = document.getElementById('board')

   if (file) {
      const fileName = file.name;
      // get the extension
      const fileExtension = fileName.split('.').pop().toLowerCase();
      fileNameElement.textContent = 'Selected File: ' + fileName;  // this automatically updates it
   
      if (fileExtension === 'mp4' || fileExtension === 'hevc') {  // good to upload
         errorElement.textContent = '';
         successElement.textContent = 'Video received. Sending for processing...';
         boardData = await sendVideoToBackend(file);  // pauses function to wait for the returned promise. await retrieves the Promise.resolve("this part")
         console.log("hello hello");
         updateBoardOnScreen(boardElement, boardData)
      }
      else {
         errorElement.textContent = 'Not a video';
         successElement.textContent = '';
      }
   }
   else {
      fileNameElement.textContent = 'No file selected';
   }
   this.value = ''  // for next file
});

async function sendVideoToBackend(file) {
   // FormData is part of js stdlib. it is an easy way to construct key:value data to send
   const formData = new FormData();
   formData.append('video', file)

   // fetch api is a way to send http requests and process responses all in one.
   // it returns a promise
   try {
      const response = await fetch('http://localhost:5000/upload', {
         method: 'POST',
         body: formData,
      });

      if (!response.ok) {  // will fail if malformed json, response doesnt exist, or response isnt ok
         throw new Error(`HTTP error! Status: ${response.status}`);
      }

      const data = await response.json();
      console.log('Response from server:', data);
      alert('Video uploaded successfully: backend says ' + data.message);
      return data.message;
   }
   catch (error) {
      console.error('Error:', error);
      alert('error occurred while uploading video. server is probably not running');
   }
}

// if i wanna show it on screen, I need another <p> and do the const element, element.textContent = thing
function updateBoardOnScreen(boardElement, boardData) {
   const columns = boardData[0].length
   boardElement.style.gridTemplateColumns = `repeat(${columns}, 63px)`;  // css

   boardElement.innerHTML = '';
   boardData.forEach(row => {
      row.forEach(letter => {
         const letterElement = document.createElement('div');
         letterElement.textContent = letter;
         boardElement.appendChild(letterElement);
      })
   })
}
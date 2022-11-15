import React from 'react';
import { useDropzone } from "react-dropzone";
import { ScaleLoader } from 'react-spinners';

const Upload = ({accept, onDrop, image, clearResult, loading, sendRequest}) => {
  const { getRootProps, getInputProps, isDragActive } = useDropzone({accept, onDrop});

  return (
    <div className='flex'>
      {!image 
      ? <div {...getRootProps({ className: "mx-auto flex justify-center items-center w-72 h-72 rounded-md outline-2 outline-dotted outline-offset-2 outline-teal-500 border-2 border-gray-300 cursor-pointer" })}>
          <input className="input-zone" {...getInputProps()} />
          <div className="text-gray-400 text-sm text-center select-none">
            {isDragActive ? "Release to drop the image here" :"Drag & Drop, or Click to select an image"}
          </div>
        </div>
      :	<div className={`mx-auto flex justify-center items-center rounded-md outline-2 outline-dotted outline-offset-2 outline-gray-400`}>
          <img 
            src={image} 
            alt="ImagePreview" 
            className={`w-72 h-72 rounded-md border-2 border-teal-500 cursor-alias uploaded-image`}
            onClick={clearResult}
          />
          {!loading
          ? <button 
              className={`absolute cursor-pointer bg-teal-500 hover:bg-red-500 text-white font-bold py-2 px-4 rounded-md predict-button`}
              onClick={sendRequest}
            >
              Predict
            </button>      
          : <ScaleLoader color="#e1e1e1" loading={loading} cssOverride={{position: 'absolute'}}/>}
        </div>
      }
    </div>
  );
}

export default Upload;
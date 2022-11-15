import React from 'react';

const Cams = ({images}) => {
  
  return (
	<div className="mx-auto mt-10 lg:mt-0 flex items-center w-full h-72 bg-gray-100 rounded-md shadow-xl outline outline-2 outline-offset-2 outline-teal-500 overflow-x-auto no-scrollbar">
    {images
    ? (images.map((image, index) => (
      <img 
        key={index}
        className="w-80 mx-5 hover:scale-110 transition-all duration-300"
        src={`data:image/png;base64,${image}`}
        alt="CAM" 
      /> 
    )))
    : <div className="mx-auto text-gray-400 text-xl">Class Activation Map</div>
    }
	</div>
  )
}

export default Cams;
import React from 'react';
import Slider from "react-slick";
import {BsFillArrowLeftSquareFill, BsFillArrowRightSquareFill} from 'react-icons/bs';
import 'slick-carousel/slick/slick.css';
import 'slick-carousel/slick/slick-theme.css';

const SamplePrevArrow = ({className, onClick}) => {
  return (
    <BsFillArrowLeftSquareFill
      className={className}
      style={{display: "block", color: "red"}}
      onClick={onClick}
    />
  );
}
const SampleNextArrow = ({className, onClick}) => {
  return (
    <BsFillArrowRightSquareFill
      className={className}
      style={{display: "block", color: "red"}}
      onClick={onClick}
    />
  );
}

const Cams = ({images}) => {
  const settings = {
    className: "center",
    centerMode: true,
    infinite: true,
    centerPadding: "40px",
    slidesToShow: 4,
    speed: 500,
    slidesPerRow: 1,
    nextArrow: <SampleNextArrow />,
    prevArrow: <SamplePrevArrow />
  };
  return (
    <div className="w-carousel text-center py-10">
      <div className="italic hover:not-italic text-gray-700 text-2xl">What the model looks at!</div>
      {images
      ? 
      <Slider {...settings} className="m-auto">
        {images.map((image, index) => (
          <div key={index} className="flex text-center cursor-pointer p-2">
            <img 
              key={index}
              className="w-72 mx-auto rounded-md hover:scale-105 transition-all duration-300"
              src={`data:image/png;base64,${image}`}
              alt="CAM" 
            /> 
            <div className="text-gray-400">{index}</div>
          </div>
        ))}
      </Slider>
      : <div className="w-48 mx-auto text-gray-500 mt-14 p-2 rounded-md border-4 border-double border-teal-500">Class Activation Maps</div>
      }
    </div>
  )
}

export default Cams;
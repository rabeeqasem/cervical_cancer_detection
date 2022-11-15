import React, { useEffect, useCallback, useState } from 'react';
import Upload from '../../components/Upload';
import Chart from '../../components/Chart';
import Cams from '../../components/Cams';
import http from "../../http-common";
import Result from '../../components/Result';

const Prediction = ({title}) => {
  const [image, setImage] = useState(null);
  const [filename, setFilename] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState({});

  const onDrop = useCallback((acceptedFiles) => {
    acceptedFiles.map((file) => {
      setFilename(file.name);
      const reader = new FileReader();
      reader.onload = function (e) {
        setImage(e.target.result);
      };
      reader.readAsDataURL(file);
      return file;
    });
  }, []);

  const sendRequest = () => {
    setLoading(true);
    const formData = new FormData();
    formData.append('image', image)
    formData.append('filename', filename)

    return http.post("predict", formData, {
      headers: {
        "Content-Type": "multipart/form-data"
      }
    }).then((response) => {
      setResult({
        prediction: response.data.preds, 
        maximum: response.data.maxProb, 
        probabilities: response.data.probs,
        images: response.data.images
      });
      setLoading(false);
    });
  }

  const clearResult = () => {
    setImage(null);
    setResult({});
  }

  useEffect(() => {
    document.title = "DrCADx | " + title || "";
  }, [title]);

  return (
    <div>
      <div className="mb-4 grid grid-cols-1 lg:grid-cols-3 ">
        <Upload onDrop={onDrop} accept={"image/*"} image={image} clearResult={clearResult} loading={loading} sendRequest={sendRequest}/>
        <Result result={result}/>
        <Chart probabilities={result.probabilities}/>
      </div>
      <div className="mt-12">
        <Cams images={result.images}/>
      </div>
    </div>
  )
}

export default Prediction;
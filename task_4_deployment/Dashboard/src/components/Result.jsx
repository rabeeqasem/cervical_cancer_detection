import React from 'react';


const Result = ({result}) => {


  return (
    <div className="mx-auto w-80 h-72 mt-10 lg:mt-0">
        <div className="my-2 flex justify-between text-gray-800 select-none">
            Prediction: 
						{result.prediction 
						? <span className="px-2 bg-green-400 rounded-lg text-white">{result.prediction}</span>
						: <span className="text-gray-400">-------</span>
						}
        </div>
        <div className="my-2 flex justify-between text-gray-800 select-none">
            Probability: 
						{result.maximum 
						? <span className="font-bold text-sm text-gray-500">{result.maximum} %</span>
						: <span className="text-gray-400">-------</span>
						}
        </div>
				<hr className="mb-2"/>
				{result.prediction && 
				<div className="h-48 text-gray-700 text-justify overflow-y-auto no-scrollbar">
					{{  'LSIL': `LSIL refers to morphologic changes along the lower end of the spectrum of SIL. About 1.7% of 
						all PAPs are interpreted as LSIL, the majority of which (>80%) are positive for HR-HPV. One of 
						the major cytomorphologic and an easily identifiable feature of LSIL is koilocytosis. Koilocytes 
						show raisinoid nuclei with sharply delineated perinuclear cytoplasmic clearing with irregular 
						outline including focal angulations. The nuclei show nuclear enlargement, hyperchromasia, and 
						nuclear membrane irregularities. Binucleation and multinucleation are frequent`,
						'HSIL': `HSIL refers to morphologic changes associated with higher end of the SIL spectrum and includes 
						both CIN 2 and CIN 3 (with CIS). About 0.3% of all PAPs are interpreted as HSIL, almost all 
						(95%) of which are HR-HPV positive. HSIL has a higher rate of progression to cancer and a 
						lower rate of regression. Long-term progression to invasive cancer is estimated at 30% for 30 years.`,
						'ASC-US': `ASC-US is relatively more prevalent with cytomorphological features revealing higher nuclear 
						atypia than reactive changes. The atypical features are equivocal for definitive dysplasia due 
						qualitative reasons with or without quantitative limitations such as atypical changes only in scant 
						cells.`,
						'ASC-H': `ASC-H is designated for cases with cellular changes that are equivocal for high-grade dysplasia 
						because of either qualitative or quantitative limitations. An ASC-H is a significant interpretation. 
						Proper designation of ASC-H cases is necessary as it might be downstream to a completely 
						different route of management.`,
						'NILM': `The category “negative for intraepithelial lesion or malignancy” is used for speci- mens that 
						show a spectrum of nonneoplastic changes, including those associated with protective and 
						reactive responses to inflammation, hormonal alterations, and colonizing or infectious organisms.`,
						'SCC': `As defined in the 2014 WHO terminology, squamous cell carcinoma is “an inva- sive epithelial 
						tumor composed of squamous cells of varying degrees of differen- tiation”`
					}[result.prediction]}
				</div>
				}
    </div>

  )
}

export default Result
import React from "react"

const introduction = () => {
  return (
    <div style={{marginTop:'30px', marginBottom : "80px"}}>
      <div style={{fontSize:'var(--text-lg)',marginBottom : "30px",fontWeight:'bold'}} >블로그 소개</div>
      <div style={{fontSize:'var( --text-md)'}} >
        <div style={{marginBottom : "20px"}}>
          머신러닝을 활용해 비즈니스 문제해결 역량을 기르기 위한 개인 커리큘럼을
          운영 중에 있습니다. 
        </div>  
        <div>
          이 블로그는 공부한 내용을 보기 좋게 정리하여 향후 복습용으로 활용하기 위해 만들었습니다.
        </div>
      </div>
    </div>
  )
}

export default introduction

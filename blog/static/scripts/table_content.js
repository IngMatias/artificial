const tableContentItemsUL = document.querySelector('.table-content-items')
const tableContentItems = document.querySelectorAll('.table-content-item')

const observer = new IntersectionObserver(changeActiveAnchor)

for (let i in tableContentItems) {
  const tableContentItem = tableContentItems[i]
  const titleText = tableContentItem.innerText
  if (!titleText) {
    continue
  }
  const id = titleText.toLowerCase().replaceAll(' ', '-').replaceAll('.', '') + `-${i}`

  tableContentItem.setAttribute('id', id)
  tableContentItem.setAttribute('data-index', i)
  
  const li = document.createElement('li')
  tableContentItemsUL.appendChild(li)
  const a = document.createElement('a')
  li.appendChild(a)
  
  a.innerText = titleText
  a.setAttribute('href', `#${id}`)
  a.classList.add('table-content-anchor')

  observer.observe(tableContentItem)
}

let activeIndex = 0
const tableContentAnchor = document.querySelectorAll('.table-content-anchor')
tableContentAnchor[activeIndex].classList.add('active')

function changeActiveAnchor(entries) {
  for (let i=0; i<entries.length; i++) {
    
    const entry = entries[i]
    const actualIndex = parseInt(entry.target.dataset.index)
    const actualText = entry.target.innerText
    
    if (
      !entry.isIntersecting &&
      actualIndex == activeIndex &&
      activeIndex + 1 < tableContentAnchor.length
    ) {
      tableContentAnchor[activeIndex].classList.remove('active')
      activeIndex = activeIndex + 1
      tableContentAnchor[activeIndex].classList.add('active')
    }
    
    if (
      entry.isIntersecting &&
      actualIndex + 1 == activeIndex &&
      0 <= activeIndex - 1
    ) {
      tableContentAnchor[activeIndex].classList.remove('active')
      activeIndex--
      tableContentAnchor[activeIndex].classList.add('active')
    }
    
  }
}

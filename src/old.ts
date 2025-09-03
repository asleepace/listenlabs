// let totalWellDressed = 0
// let totalYoung = 0

// /**
//  * Main function where we should let person in or not...
//  */
// function shouldLetPersonIn({ status: next, metrics }: GameState): boolean {
//   if (next.nextPerson == null) return false;

//   const totalCount = 'admittedCount' in next ? next.admittedCount : 0

//   const { nextPerson } = next

//   // NOTE: count total number of people
//   if (next.nextPerson.attributes.well_dressed) {
//     totalWellDressed++
//   }
//   if (next.nextPerson.attributes.young) {
//     totalYoung++
//   }

//   // calculate totals

//   if (totalYoung > 600 && totalWellDressed > 600) {
//     return true
//   }

//   if (nextPerson.attributes.well_dressed && nextPerson.attributes.young) {
//     return true
//   }

//   if (totalYoung < 595 && nextPerson.attributes.young) {
//     return true
//   }

//   if (totalWellDressed < 595 && nextPerson.attributes.well_dressed) {
//     return true
//   }

//   if (totalCount > 975) {
//     if (totalYoung < 600 && nextPerson.attributes.young) return true
//     if (totalWellDressed < 600 && nextPerson.attributes.well_dressed) return true
//     return false 
//   }

//   const hasOneOrMoreAttribute = nextPerson.attributes.well_dressed || nextPerson.attributes.young 

//   return hasOneOrMoreAttribute || nextPerson.personIndex % 8 === 0
// }

/** Bind food identity to immutable native geom IDs. Position projections must
 * not turn contact with the bowl below a fruit into contact with the fruit.
 */
export function createContactFoodResolver(mj,model,fruit,fruitGeomNames){
  const foods=new Map();
  for(const [name,index]of Object.entries(fruitGeomNames)){
    const id=mj.mj_name2id(model,5,name); // mjOBJ_GEOM
    if(id<0||model.geom_bodyid[id]!==0||!fruit[index])throw new Error(`Invalid fruit contact geometry: ${name}`);
    foods.set(id,fruit[index]);
  }
  return id=>foods.get(id)??null;
}

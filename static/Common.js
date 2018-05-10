/*
* author : Jian-Kai Wang (http://jiankaiwang.no-ip.biz)
* project : seed 2016
* github : https://github.com/jiankaiwang/seed
*/

var getDictionaryLength = function(getDictObj) {
	var dictLength = 0;
	for (var key in getDictObj) {
			if (getDictObj.hasOwnProperty(key)) {
					dictLength += 1;
			}
	}
	return dictLength;
}

var getDictionaryKeyList = function(getDictObj) {
	var keyList = [];
	for(var key in getDictObj) {
		keyList.push(key);
	}
	return keyList;
}

var getDictionaryValueList = function(getDictObj) {
	var valueList = [];
	for(var key in getDictObj) {
		valueList.push(getDictObj[key]);
	}
	return valueList;
}

var dictBubbleSortOnValue = function(getDictObj) {
	var retKeyList = getDictionaryKeyList(getDictObj);
	var tmpKey = "";
	// sort body
	for(var i = 0 ; i < retKeyList.length-1 ; i ++) {
		for(var j = 0 ; j < retKeyList.length-1-i ; j++) {
			if(parseFloat(getDictObj[retKeyList[j]]) > parseFloat(getDictObj[retKeyList[j+1]])) {
				tmpKey = retKeyList[j];
				retKeyList[j] = retKeyList[j+1];
				retKeyList[j+1] = tmpKey;
			}
		}
	}
	return retKeyList;
}

/*
 * desc : sort keys in dictionary by their values
 * para : 
 *	1. getDictObj : { key : value }
 *	2. sortType : "bubble"(default)
 * 	3. getOrder : desc, asc(default)
 *	4. getListCount : 0-N
 * example :
 * 	var aa = { 'a' : 10, 'b' : 3, "c" : 5 }
 *  var keyList = getKeyBySortOnValue(aa, "bubble", "desc", getDictionaryLength(aa));
*/
var getKeyBySortOnValue = function(getDictObj, sortType, getOrder, getListCount) {
	var retKeyList = getDictionaryKeyList(getDictObj);
	switch(sortType) {
		default:
		case "bubble":
			retKeyList = dictBubbleSortOnValue(getDictObj);
			break;
	}
	// getOrder : desc, asc
	var tmpKeyList = [];
	switch(getOrder) {
		case "desc":
			for(var i = retKeyList.length-1 ; i >= 0 ; i--) {
				tmpKeyList.push(retKeyList[i]);
			}
			retKeyList = tmpKeyList;
			break;
	}
	// return as desired number
	tmpKeyList = [];
	var keyLength = getListCount > getDictionaryLength(getDictObj) ? getDictionaryLength(getDictObj) : getListCount;
	for(var i = 0 ; i < keyLength ; i++) {
			tmpKeyList.push(retKeyList[i]);
	}
	retKeyList = tmpKeyList;
	return retKeyList;
}

/*
 * desc : return all index in the list which their values are the same with the given value
 * inpt :
 * |- getList : the searching list
 * |- getItem : the searching item
 * retn : a list containing the index
 * e.g. : allItemIndexinList([1,2,3,3,4], 3);  // return [2,3]
 */
var allItemIndexinList = function(getList, getItem) {
	var retList = [];
	var count = 0;
	
	getList.forEach(function(data) {
	  if(data == getItem) {
			retList.push(count);
		}
		count += 1;
	});
	
	return retList;
}

/*
 * desc : return the list which elements are duplicated
 * inpt :
 * |- getList : the list 
 * retn : the list with the non-duplicated element
 */
var uniqueList = function(getList) {
	var retList = [];
	for(var i = 0 ; i < getList.length ; i++) {
		if(retList.indexOf(getList[i]) < 0) {
			retList.push(getList[i]);
		}
	}
	return retList;
}

/*
 * desc : count byte length of the UTF8 string
 * inpt :
 * |- getStr : a UTF-8 encoding string
 * retn : a number
 */
var byteCount = function(getStr) {
    return encodeURI(getStr).split(/%..|./).length - 1;
}





Clustering The New Yorker Caption Contest Submissions with k-means
==================================================================
by Charlie Hack <charlie@205consulting.com>


This script uses scikit-learn to cluster Caption Contest submission documents
by topics, using a bag-of-words approach. 

There are two methods available:

  - TfidfVectorizer uses a in-memory vocabulary (a python dict) to map the most
    frequent words to features indices and hence compute a word occurrence
    frequency (sparse) matrix. The word frequencies are then reweighted using
    the Inverse Document Frequency (IDF) vector collected feature-wise over
    the corpus.

  - HashingVectorizer hashes word occurrences to a fixed dimensional space,
    possibly with collisions. The word count vectors are then normalized to
    each have l2-norm equal to one (projected to the euclidean unit-ball) which
    seems to be important for k-means to work in high dimensional space.

It's also possible to transform the corpus with Latent Semantic Analysis before
applying the clustering, though empirically this doesn't seem to show a marked
improvement in the quality of the clusters.

It's hard to measure quantitatively how well the algorithm is doing at categorizing,
so it's easiest to just play around with different values for k and see what works.
Luckily the datasets are small (< 10000 documents) so that the script runs fast.

This is inspired by the scikit-learn documentation by Peter Prettenhofer and Lars 
Buitinck.


Sample output for this cartoon:

![Winning caption: "The sign said 'Home-Style Cooking.'"](http://legacy.newyorker.com/images/2014/10/13/p323/141013_contest_p323.jpg)



```
loading docs...
4324 documents.

Extracting features from the training dataset using a sparse vectorizer...
done in 0.142360s.
n_samples: 4324, n_features: 1476

Clustering sparse data with KMeans(copy_x=True, init='k-means++', max_iter=100, n_clusters=20, n_init=1,
    n_jobs=1, precompute_distances=True, random_state=None, tol=0.0001,
    verbose=True)...
Initialization complete
Iteration  0, inertia 7037.315
Iteration  1, inertia 3882.707
Iteration  2, inertia 3840.816
Iteration  3, inertia 3820.713
Iteration  4, inertia 3807.901
Iteration  5, inertia 3801.468
Iteration  6, inertia 3798.675
Iteration  7, inertia 3794.652
Iteration  8, inertia 3791.420
Iteration  9, inertia 3789.638
Iteration 10, inertia 3787.505
Iteration 11, inertia 3785.082
Iteration 12, inertia 3782.879
Iteration 13, inertia 3780.864
Iteration 14, inertia 3780.040
Iteration 15, inertia 3780.012
Iteration 16, inertia 3779.980
Iteration 17, inertia 3779.948
Iteration 18, inertia 3779.928
Iteration 19, inertia 3779.916
Converged at iteration 19
done in 0.557s.

cluster 1:
===========
Hot Date
What's hot here?
Are your feet hot?
"It's a hot spot."
Is it hot in here?
Are your legs hot?
"I think you're hot"!
I hope the soup is hot.
stove or not you are hot
"Finally, hot pancakes!"
"Best hot meal in town."
"Your plate will be hot."
It's getting hot in here.
"This is going to be hot!"
Cooking, so hot right now.
Let's have the table d'hot!
"They call it 'Hot Cuisine'."
I just simple love hot meals.
I hope the food's hot enough.
"It's going to be a hot date!"
This one might too hot for me.
I'll have the hot plate special
This place is so hot right now.
Your burner is never hot enough
I heard this restaurant was hot.
Think I'll go with the hot dish.
I suggest avoiding the hot dish.
"This is going to be a hot date."
Every item is a Hot Plate Special.
"I'll have the hot plate special."
OK, no more hot new restaurants...
Are your legs hot? Mine are baking!
Is it just me or is it hot in here?
"I'm having the hot-plate special."
He said their #3 is very, very hot.
Is it hot in here or is it just me?
"At least the plates should be hot"
So this is your idea of a hot date?
I hear the food here is really hot!
"I guess we're going 'Table d'Hot'?"
Is it hot in here, or is it just me?
Is it just me, or is it hot in here?
"This wasn't my idea of a hot date."
Is it hot in here, or is it just me?
Is it just me, or is it hot in here?
Did I ever tell you how hot you were?
I guess we have to order a hot plate!
"The Hot Plate Special is really hot"
"So this is your idea of a hot date?"
Cool. These guys are getting hot pot.
Everything here is hot off the stove.
I think I'll just get the hot pockets.
Go ahead. Touch it and see if it's hot.
I hear this is the new hot spot in town
Well, I said this place was a hot spot.
"They tell me this place is really hot"
Interactive dining is so hot right now.
I think I'll have the Hot Plate Special.
So this is what you meant by a hot date??
I warn you. The cuisine here is quite hot.
I think I'll have the 'Hot Plate Special'.
This Hot Stove League has really expanded.
"I'm thinking something hot off the grill."
"I thought hot stove referred to baseball."
They say the Hot Plate is excellent tonight.
The food here is hot, but it's not that hot.
I don't see why this restaurant is so hot....
Wow - this could turn into a really hot date!
"It looks like a hot stove league night out."
And you didn't want to slave over a hot stove!
"At least we're not standing over a hot stove."
Careful, I hear the plates here are really hot.
"hot-plate dating" seemed so extreme at first..."
...it sure beats slaving over a hot stove at home
This was not what I meant by having a "hot date".
OK, I get it. Hot flashes are no fun. Can we go now?
The only thing on the menu is the hot plate special.
when they say "hot off the stove" they're not kidding
Hey baby everything is hot! Hot soup? No hot prices !
Poor Sandra, stuck at home slaving over a hot stove...
It's getting hot in here, so take off all your clothes.
I'm torn between "Hot and Fresh" and "Have it Your Way".
At least you can still purchase a hot meal back in Coach.
Do we get our money back if this doesn't make us both hot?
Honey, the critics were right. These tables really are hot!
But I thought you didn't want to slave over a hot stove tonight...
"When the ad said, 'food guaranteed hot', I knew there was a catch."
This was supposed to be haute cuisine, not do-it-yourself hot cusine!
"See honey, I told you stove-top table restaurants make for a HOT date!"
Well, I must say this is not what I expected at authentic Chinese hot pot
"Now isn't this better than you stuck cooking in a hot kitchen all night?"
Now, isn't this better than slaving over a hot stove for your anniversary?
"I can't decide between the stove top casserole or the hot plate special."
You have been slaving over a hot stove all day so I thought I would take you out.
Of course, the "slaving over a hot stove" complaints will never work again, you know..
"Says here they have a special on Thursdays where you can make your soup in a hot tub."
I like the "serve it piping hot" part but hate having to "simmer it gently for six hours"
How irronic they should charge so much for us to cook our own food. Is it hot in here to you?
"Since you never cook anymore I thought you might find it intriguing to sweat over a hot stove."
Hot cross buns, hot toddy, hot potato... What does a guy have to do to get something cold around here?
Natural Marriage in the 'Hot Seat' of public ridicule by the Supreme Court will always remain an institution of love.
I know we agreed to share everything but is there really any point to us both sweating over a hot oven every Saturday night?

102 captions.


cluster 2:
===========
The menu says DIY,
Whats menu flambe?
My menu is cooking.
Hey, the menu's blank.
Says their menu ranges.
Lets order off the menu.
The menu feels ambitious.
Great, the menu is blank.
"Let's get the raw menu."
"This menu is half-baked."
I like the full-range menu.
Let's ask for the oven menu.
My menu is a seed catalogue.
"This menu is from Benihana"
Honey, it's not on the menu.
The menu covers a wide range.
I think your menu is on fire.
Can we get a beginner's menu?
This DIY menu is confounding.
Let's check the electric menu!
This is a cookbook not a menu.
"Honey, your menu is smoking."
It’s an ‘A la cooker’ menu
This menu is just a grocery list.
"They don't have toast on the menu"
This isn't a menu, it's a cookbook!
Look at the 'cook it yourself' menu!
"I don't see dishwasher on the menu"
Where on the menu can I order a chef?
"Do you think there's a takeout menu?"
the menu is old i can not eat honey!!!
I can't find sushi on the menu either.
Why are"Grilled Kneecaps" on the menu.
This 'menu' looks eerily like a recipe.
Hey! This menu is just a grocery store!
Hey, this isn't a menu. It's a cookbook!
Hurray, they have Jiffy Pop on the menu!
The menu notes "Some assembly required."
"Oh look, they have souffle on the menu!"
There's never anything new on the menu...
The menu offers a range of possibilities.
Well, the menu does say "fresh cooked"...
What portion of the menu are you reading ?
Stuffiing's on the Stove Top's menu again!
i do not see hotcakes anywhere on the menu
I think I'm gonna go with the tasting menu.
"It's not really a menu. It's a recipe card."
The menu says it's an energy efficient model.
"I guess there is nothing Paleo on the menu".
The menu says "they serve food on hot plates".
I think I'll order the pan-seared chef's menu.
"They have a wide range of items on the menu."
Long menu. Oughta be called Tome on the Range.
I see they took Stovetop Stuffing off the menu.
Looks like the new owner changed the menu. Ouch!
My menu only has grilled cheese and frozen pizza!
It's not a menu.It's a first aid guide for burns."
So, do you see something on the menu you can cook?
Everything on this menu makes me have to take a shit.
Think I'll stick with the vegan menu... just in case.
Should we tell the waiter which menu we brought along?
Now I understand the "warm and cozy" claim on the menu
"My menu choice look like it's from some back-burner."
"You're right. The menu doesn't list 'micro heuvos' ".
" Do You Have The Cook On High Or Medium Or Low Menu? "
Does everything in your menu start with the word burnt?
"There's nothing on this menu that I know how to cook."
"Ugh. Is there anything on this menu that isn't fried?!"
The best dish on the menu is actually the steak tartare.
I was thinking sushi, but I'm not finding it on the menu.
If you eat your vegetables, they bring you a dessert menu.
You pick the wine. I'm still looking at the insurance menu.
I guess that's what they mean by a "range" of menu options.
"It's the hottest new restaurant in town. Watch your menu!"
"Good luck here honey. This menu has almost all baked food."
"Sorry, dear, but I prefer the micorwave restaurant's menu."
I don't think you're supposed to put you menu on the burner.
"Have you checked out this menu? There's a whole range here."
It's your birthday, so feel free to cook anything on the menu.
"Wow, 'It's Just Lunch' has added cooking dinner to its menu."
Strange, I got a recipe book instead of a menu. What about you?
I've heard not to order anything on the menu that's "blackened"
"They decided to do something about picky-picky menu requests."
Let's just order matches. Says today's special is "flaming menu."
Sweetheart, careful with the menu. Your burners may still be hot.
I don't see your burnt soup on the menu, shall we order it anyway?
Their liability insurance disclaimer is on the bottom of the menu.
it's not a menu - just something about the range-to-table movement
"The only thing on this menu I know how to make is the boiled egg."
"Please listen closely, as our menu options have recently changed."
What a great menu. It looks like everything they offer is "homemade".
I don't think that the menu needed to say that everything is homemade.
If we're not careful everything on this menu is going to be well done!
Why did you pick a Thai restaurant? I can't make anything on this menu.
"Gee honey,why is everything on the menu only baked,fried or broiled?''
If we move the issue to the front burner, it will incineratre your menu.
"The menu says if we bend over our stove top we will cook our pop eyes."
And how do you think the menu when fried would taste with a little salsa?
The menu says to keep your elbows off the table while the burners are on.
Jim told me he had the oven roasted nuts, but I don't see it on the menu.
"I'm not super hungry tonight: do you want to just do the two burner menu?"
Why does my menu say that everything can be served raw or burned to a crisp?
"What's on your menu dear? Kinda resembles Ikea directions to me sweetheart."
let's get through this hold harmless agreement and get the menu, I'm starving!
Same menu classics, now made fresh at your table. So, how long will this take?
I've been here before. Two tips: Mind your elbows, and don't set your menu on fire.
As Ed glanced over the menu at Carol, he was certain things would soon be heating up.
"They say it's the new spin on 'Self-Serve.' Whoa! Helen hold that menu a little higher."
whoa this place really has some hot menu items, be careful what you order you could get burned
The menu consists of four elements. There's no salad, but otherwise the range of options is O.K.
It suddenly became clear why the only items on the menu were Stove Top stuffing and Jiffy Pop popcorn.
"Don't worry. They'll bring the ingredients and the cooking instructions once we order from the menu."
See, it says it again at the bottom of the menu, "cooked entirely to your satisfaction or your money back " !
"Honey, I think our table has a gas leak." "No, I'm still farting. Shhh. Just keep looking at the menu. Shhhh. "
Look, honey- the menu says we could have Gordon Ramsay or your mother watch our table and then insult our cooking.
"Considering the hot seat the cartoonist put us in this time, I say we order the most expensive wine on the menu."
"Is your's a menu. Mine is only the operating, maintenance and repair instructions and a schedule of the gas charges."
I can appreciate the homestyle ambiance of this place but the menu choices are just disgustingly limited. Let's get out of here.
Hey Marge, could you please read me the disclaimer on the bottom of the menu -something where it says the management won't be held responsible for anyone with long dangling neckties, etc., etc. .

119 captions.


cluster 3:
===========
"You love cooking."
So...who's cooking?
Shall we get cooking?
Who's cooking tonight?
self-cooking restaurant.
Am I cooking or are you?
Hey! What's cooking baby?
So, what are you cooking?
I'll have what he's cooking
I'll have what he's cooking.
"I want what they're cooking."
"I'll have what he's cooking."
"But it's still your cooking."
It's better than cooking alone.
So what are you cooking tonight?
"I'm cooking what he's cooking."
"I'll have what they're cooking."
Who's cooking tonight, you or me?
What are you cooking tonight dear?
"Are you cooking tonight or am I?"
It says cooking utensils are BYO."
The sign did say Homestyle Cooking.
I'll have what you're cooking, dear.
Fine Dining? More like fine cooking!
Tableside cooking was just too passe.
But you said you were sick of cooking!
And I came here to escape your cooking.
We're not cooking the bacon on my side.
...but I thought you were tired of cooking?
Everyone is cooking for themselves tonight.
When I finish cooking do I serve the waiter?
I didn't want this kind of homestyle cooking
"They wern't kidding about homemade cooking"
The Yelp review says your cooking is terrible
I know, dear, but I'll do the cooking tonight.
I'd bet that this cooking show gets cancelled.
So, your Match profile said you enjoy cooking.
Wait, is he serious about cooking our own food?
This isn't what I meant by "homestyle cooking."
“ It’s called Cooking. Got great reviews.”
"It's called a stove and it's used for cooking".
At last something we can agree on: cooking time.
I thought you said self serving not self cooking.
This wasn't what I meant by 'you do the cooking'.
Steep prices considering we are doing the cooking!
Listen Richard, It may be chic, but I'm not cooking
"And who,pray tell, is supposed to do the cooking?"
Let's start off with ordering some cooking utensils.
Happy Anniversary Honey. I'll do the cooking tonight.
i guess you could interpret this as homestyle cooking
Speed cooking... need to move to the next table soon.
Honey, I hope this doesn't mean you're cooking dinner.
Since when has dining out meant cooking your own food?
I thought you could use a break from cooking all week.
"Happy birthday my dear, for once I'll do the cooking."
This counts as double points, dinner out and me cooking.
It's the first date and e already has me cooking for him
Is this your idea of away-on-the-range stove top cooking?
"I can't do this. There aren't any cooking instructions."
So, tonight I'll do the cooking and you can do the dishes.
"Shall we start with a nice 2004 bottle of cooking sherry?"
Well the review said the minimalist cooking is spectacular.
"They find that this system eliminates cooking complaints."
Since we have to do the cooking, do we have to leave a tip?
My Boy Scout open campfire cooking skills will be handy tonight.
"You're Right, Cooking Out is Much More Romantic Than Cooking In"
"This is not what I pictured when Zagat said 'homestyle cooking.'"
I can’t decide between under cooking chicken or burning meatloaf.
Ironically, this evening was devised as an escape from your cooking.
Now that we're here, I'll ask you again: whatcha cooking for dinner?
"All I'm saying is cooking on a stove is not a new concept in dining!"
My therapist said cooking you dinner would impress you on a first date.
Now I can insult the cooking and not worry about my food being spat on.
"What's cookin' good lookin'?" "But really...What do you have cooking..."
Now I understand why the cooking here always gets such good online reviews.
If you think cooking it yourself is weird, wait till you see the restrooms.
This is not what I meant when I told you I need a break from cooking dinner!
When they said "Guaranteed homestyle cooking", I should have seen this coming.
I remember when 'going out for dinner', meant I didn't need to do the cooking.
Fred would never learn he misunderstood 'interest in cooking' on Kim's profile.
How much is this dinner when we're cooking it ourselves, eating out got weird...
You did say you were cooking dinner but I didn't think reservations were necessary.
I should warn you. Consuming my cooking may increase your risk of food borne illness.
This is so crazy, who would have thought that cooking your own food would be a thing.
"Sure, you do the cooking and washing up, but it's 'all you can eat' - and no tipping!"
"The review said 'stove-top cooking, like mom used to make,' but I didn't expect this."
Darn, I left my food handler's permit at the office...would you mind cooking tonight, hon?
Your cooking is great, honey, but even you'll admit it's nice to eat out every once-in-a-while.
"Homey," dear? Really? Wouldn't that imply that included among your many talents is some semblance of cooking?
The "If you know so much about freaking cooking then do it your dam self " Quail avec Pomme Frites should be fairly simple.
I'm beginning to think that the participatory performance art of cooking is eclipsing the participatory performance art of eating.

91 captions.


cluster 4:
===========
Salad?
"no salad bar" ?
I'll have the salad.
"The salad's safer."
I'll have the salad.
The salad looks good.
"What? No salad bar?"
"I just want a salad."
"I'll have the salad."
I always get the salad
I just wanted a salad.
I recommend the salad.
"Wanna share a salad?"
Let's just order salad.
Yeah, salad for me too.
"Let's skip the salad."
"You want a ....salad?"
"The salad looks good."
I really wanted a salad.
I'm just having a salad.
"I'll just have a salad."
I'll just have the salad.
I think I'll have a salad.
I think I'll have a salad.
I think I'll have a salad.
I think I'll have a salad.
I recommend the Cobb salad
Let's just have the salad.
"We need a new salad bar."
I think I will have a salad
I think I'll order a salad.
"Let's just order a salad."
"Just salad for me, thanks."
I think I'll have the salad.
I think I'll have the salad.
I think I'll have the salad.
"I think I'll order a salad,"
I am going to have the salad.
I think I will get the salad.
I think I'll just do a salad.
"Just a salad for me, thanks."
I'm just going to get a salad.
I think I'll just get a salad.
I think I'll just get a salad.
Their specialty is cobb salad.
"Funny tables for a salad bar."
I think I'll just have a salad.
I think I'll just have a salad.
I think I'll just have a salad.
I'll have a salad and gazpacho.
I think I'll just have a salad.
"I think I'll just get a salad."
"I think I'll just get a salad."
I think I'll just have a salad .
"I think I'll just have a salad."
"I think I'll just order a salad."
What'd you expect after salad bars?
Maybe we should just order a salad.
I think i'll just have the salad bar.
"The cold soup and a salad, I think."
On second thought, I'll have a salad.
I think I'm just going to get a salad.
"Shucks. I was hoping for Cobb Salad."
"I think I'm going to have the salad."
"The iceberg endive salad looks good."
I know, but I really just want a salad.
" I was thinking. Maybe just a salad ?"
I think I'm just going to have a salad.
I think I'm just going to have a salad.
I think I'm just going to have a salad.
"I think I'll just order the salad bar."
How wrong would I be to just have a salad?
I'll have the salad and a bowl of gaspacho.
"Everything is good here except the salad."
On second thought, I'll get the salad, too.
Well, as usual, I'll have the nicoise salad.
"Darn, I was really in the mood for a salad."
"I'll have a steak tartare and a Caesar salad."
Stove top this, stove top that. I just want a salad.
For some reason, I'm just feeling like a salad tonight.
I'll just have the salad tonight - it's too hot to cook.
Honey, in all seriousness I would recommend a cold salad.
I'll just order a salad. I'm not in the mood to cook tonight.
I don't know how to make any of these. I guess I'll just have a salad.
For some reason I have a fire in my belly. I'm going to just have a salad.
Just order the salad and pray we weren't supposed to bring our own lettuce!
I'll clean the fish while you go out to the garden and pick the salad greens.
"You just want a salad, too? That should've been mentioned before we picked this restaurant."

88 captions.


cluster 5:
===========
Free-range hotspot
It's hardly free range.
Everything's free range.
See anything range free?
Definitely not free range.
The free range looks good.
They only serve free-range.
" I love free range dining."
I see they serve free range.
They said it was free-range.
I think it's all free range,
So much for a free-range menu
Looks like it's all free-range
Everything here is free-range.
Everything here is free range.
"It's a free-range restaurant."
Well, so much for "free" range.
Next time, let's eat free range.
"Everything here is free range."
All the food here is free range.
Every element here is free range
"OK, so this is free-range food."
I'm having the free range chicken,
"The free range looks interesting."
"Let's just go free range tonight."
There's not much here in our range.
I hope you like free range cooking.
This isn't my idea of "free-range".
I see their meat is all free-range.
Free range cooking is kinda pricey.
Wow, even the tables are free-range.
Well this range ain't exactly 'free'
Wonderful, Entire menu is free range
Free range chicken is all they have.
It seems we have a range of options.
"The free-range chicken looks good."
Everything on the menu is free range.
Free range chicken sounds appetizing.
"I really prefer free range chicken."
Everything on the menu is free range.
"All of the entrées are free range."
"I recommend the free range chicken".
I thought free range was all the rage.
I'm going with the free range chicken.
I wonder if the chicken is free range.
"I love these free-range restaurants."
Do you think the chicken is free range?
What's free - the range or the chicken?
"I wonder if the chicken is free range?"
Told you this restaurant was free range.
I think I'll try the free range chicken!
So, this is free-range chicken. Who knew?
I think I'll have the free range chicken.
I think I'll have the free-range chicken.
I think I'll have the free range chicken.
I think I'll skip the free range chicken.
I'm curious about the free range chicken.
"I think I'll have a free range chicken."
I hear their free range chicken is decent.
"I love these new free range restaurants."
I thought free range meant something else.
I thought free range meant something else.
It's Free-Range Bring Your Own Everything.
I thought free range referred to the menu.
"I'm going to have the free range chicken."
"I think I'll have the free-range chicken."
"I'm thinking the free range chicken. You?"
"I think I'll have the free-range chicken!"
"Oh, that's what they mean by 'free range.'
Don't worry, everything here is free-range.
I understand the range chicken is delicious
It says here even the tables are free range.
"So that is what they meant by Free-Range.''
"Mmmm. We should have free-range more often."
"Technically, everything here is free range."
That's funny. All the entrees are free-range.
"Free range chicken with stove top stuffing?"
"I thought 'free range' meant something else"
"I thought 'free range' meant something else."
I guess this is the free range for the chicken.
"I thought free range was something different."
"The free-range cordon bleu here is exquisite."
Would you like "Free Range" or "Farm-to-Range"?
'Are you getting the free range chicken, honey?'
So this is what they mean by free range cuisine.
"When I called, they promised a free range menu."
I've heard their specialty is free range chicken.
"I didn't think 'free range' cooking meant this."
I thought free range dining meant something else.
How about free range chicken with solar electric?
I guess 'free range' wasn't referring to the menu.
"I guess I misunderstood what 'free range' meant."
So this is what the ad meant by "Free-Range Menu"?
This is not the sort of "free range" I had in mind.
It says here all their poultry dishes are free range
Well, the ad did say it was a free range restaurant.
" I'm having the free range chicken, what's yours? "
What do you think they mean by "Free Range Chicken?"
This is taking free range to an uncomfortable level.

[...]

cluster 6:
===========
Don't get the sushi.
" I don't see Miele "
“Don’t lean in.”
I don't see any salads.
I don't see any salads.
Don't look under the hood
"Don't ask for placemats."
I don't know what to make!
Don't tell me to simmer down.
"Don't think of it as 'slaving'."
Don't choose the Beef Wellington.
"Don't they have any cold plates?"
"I don't recommend the ice cream."
"I kind of miss chefs, don't you?"
"I hope they don't burn our food."
Don't put your elbows on the table.
says here they don't use microwaves.
Don't even think about the soufflé.
I'm sorry. I don't even own an iPot.
"At least we don't have to clean up!"
"I don't see onion volcanos on here."
Why don't they have cold dishes here?
"I don't think this is a Benihana's."
I just adore homemade food, don't you?
"I don't think we need to leave a tip."
I hope you don't expect a tip for this.
Now I know why they don't have take out.
At least we don't have to do the dishes.
At least we don't have to do the dishes.
I don't care who brought home the bacon.
At least we don't have to do the dishes.
I love these DIY restaurants, don't you?
At least we don't have to do the dishes.
I don't get it -- you don't cook at home!
I don't see calf brains and scramble eggs.
Don't you just love dinner by pilot light?
I don't care that the French are doing it.
I don't see Gazpacho on the menue anymore.
Don't take the beef. It's always overdone.
At least we don't have to clean the dishes.
Kind of home on the range, don't you think?
Ugh, we don't know how to make any of this!
"This is why chefs don't open restaurants."
Don't they have anything that's NOT flambe'?
Don't panic, but I think I forgot my mallet.
"But I don't want to meet my needs anymore."
"I'm assuming we don't have to leave a tip."
"At least we don't have to wash the dishes."
Well, at least we don't have to slaughter it.
At least we don't have to go shopping too....
Honey, I don't think you know what DIY means.
I don't know. Korean Barbecue seems dangerous.
Hopefully they don't overcook my burger again.

[...]

cluster 7:
===========
"Cook here often?"
Who's turn to cook?
I don't cook. Do you?
No, it said cook top.
Do you cook here often?
"It's all you can cook"
It's your turn to cook.
"I can't cook, can you?"
"Didn't I cook last week?"
"Tomorrow, let's cook in."
I can only cook spaghetti.
"You hate to cook at home"?
"I never know what to cook."
"I can't cook, but I will fry."
They cook it right at the table.
"If I cook, it's Pappa's pizza."
There's nothing here I can cook.
New restaurant, "You Cook" opens.
all the meals here are a la cook.
we weren't going to cook to night
Its nice to cook out occasionally
"So, what do you like.. to cook."
Are we on candid camera cook-off?
"kind of pricey for cook your own"
"How come you never cook anymore?"
I hate these cook your own places.
"The ad did say, 'Cook to order.'"
We're out to dinner so I can cook?
It's nice to cook out for a change.
Shall we go dutch and cook our own?
Honey, which of these can you cook?
"Whose turn is it to cook tonight?"
They expect me to cook on electric?
I said I didn't want a cook tonight.
It's an all you can cook restaurant.
Great night to cook out of the house
It's nice to cook out, for a change.
Do you want to cook or do the dishes?
Order something you know how to cook.
"I don't want to cook dinner at home"
Cook like there's no one watching you!
sweetheart, what did you want to cook?
Are you suggesting we should cook more?
So Barbara tells me you're a good cook?
I thought I didn't have to cook tonight!
There isn't a thing here that I can cook.
I told you I didn't want to cook tonight.
You said you'd pay to see me cook dinner!
It just says "Cook it Up; Pay it Forward"
Honey, i wish you had learned how to cook.
Can't believe I have to cook again tonight.
Well, this is awkward. I can't cook either.
I thought your profile said you could cook?
So Laura, you're gonna cook all this right?
Well, you keep telling me you love to cook.
Hmm--29 ways to cook chicken on a stovetop.
Why didn't we just stay home and cook honey?
Dear, I'm in no mood to cook! Let's go home.
"I told you that you wouldn't have to cook."
"It would be a lot cheaper to cook at home."
We order the food, cook it , and save money.
You Match profile said that you like to cook.
Everyone says that we must cook a steak here.
"What would you like to cook for me, sweetie?"
With these prices, I have to cook my own food?
You said you didn't want to stay in and cook...
"I'm glad we came here. We never cook at home."
"Cook my own balls soup?! Not at these prices!"
"If I wanted to cook I would of stayed at home."
It's the latest thing -- you cook your own meal.
We should leave. I prefer to cook with electric.
"I lied on my profile about being able to cook."
Seriously, who has time to cook at home anymore?
But you save two dollars if you cook it yourself.
Sorry, honey. You're still going to have to cook.
So Marge....20% discount if we cook it ourselves.
Well there's always the all you can cook special!
Honey, I thought you didn't want to cook tonight.
"It's our anniversary. Let's spring for the cook."
I hear the chicken piccata is really easy to cook.
Your profile said you love to cook. You're welcome.
At these prices they could at least cook it for us.
"If this is cook-your-own, why do they have menus?"
It's times like this I wish one of knew how to cook.
"It says here you can cook your food and eat it too."
"Let's make believe we're home. You cook everything."
"But when I cook at home, I don't have to wear pants"
"Well, this is ONE way to get you to cook something!"
"'Cook Your Own' is a very raw deal at these prices."
Why doesn't the cook cook because this is ridiculous?
"It's called a stove dear. It's used to cook food on."
I know it's not traditional, but I will cook the meal.
"The trick is to order something you know how to cook."
I prefer the old way over this cook-it-yourself option.
"Looks like we came on the line cook's night off again."
We're paying them for this food, shouldn't they cook it?
We should have stayed home. I can cook better than this.
You were right. Go Cook Yourself isn't just a funny name.
Didn't you mention in your profile that you LOVE to cook?
I thought we were supposed to cook tonight, not dine out.
If I order the steak, honey, can you cook it medium rare?
"It's the latest thing. You cook right at your own table."

[...]

cluster 8:
===========
"What? No microwave?"
I prefer the microwave menu.
At least it's not a microwave.
"No microwave items on the menu."
OK! You can buy the new microwave.
Where are the microwave selections?
Let's move to the microwave section.
Isn't there a microwave we can sit at?
Could we go with the microwave option?
This is so much better than microwave.
It really is better than the microwave.
Much better than the microwave section.
"Do you mind if I ask for a microwave?"
I prefer Mike's Microwave Bistro myself.
"I can't wait till thry get a microwave"
I think I'll have the microwave chicken!
It sure beats their old microwave place.
You promised me there'd be a microwave...
"Can't we just go to that microwave joint?"
Next time let's go to that microwave place.
"We should've went to the microwave place."
"Do you think we can get a microwave table?"
The microwave restaurant has faster service.
Way better selection than the microwave place.
I think we should move to the microwave table.
We should have waited for the microwave table.
"Too involved; let's go to the Microwave Room."
"I guess their microwave theme didn't make it."
For an extra $25 they'll bring out a microwave.
Well, it's an upgrade from the Microwave Café.
I'm in a hurry, is there a microwave table open?
"See anything that can be done in the microwave?"
Last time they seated us in the microwave section.
I'm in a hurry. Let's order off the microwave menu.
Home Cooking Out ... but I don't see a microwave...
This reminds me of the microwave place where we met.
I think it's time we tried that new microwave joint.
If you take a doggie bag, they lend you a microwave.
The service is so much faster at the microwave place.
This restaurant is too chic for me, I only microwave.
Boy that microwave popcorn sure does sound good . . .
On second thought, let's switch to a microwave table.
I don't know. I'm leaning toward the microwave brunch.
The show starts at eight, should we go with microwave?
"Nice to get away from the microwave once in a while."
The food is better then the place with microwave ovens.
Let's see what they've got over at a microwave table...
Next time, let's just go to Le Chateau Microwave instead.
"We should have gone to Microwave, they have a better selection"
I know you're short on time, but the microwave table was booked.
I think I got the wrong menu—this one has microwave directions.
I'm really not very hungry tonight. Let's try the microwave room.
"We could save a lot of money if we had a microwave oven at home."
The curtain is in an hour. Maybe we should ask for a microwave table.
“Since we’re in a hurry, let’s sit in the microwave section.”
Its a lot more promising than the microwave tables we ate at last week.

[...]


cluster 9:
===========
We cooked?
Home on the range
I love home cooking
"Home On The Range."
Nursing Home Kitchen
At home on the range.
"It's all home cooked!"
I am at home on the range
I just love home-cooking.
We should have stayed home.
"I love home cooked meals."
"I just love home cooking."
I prefer home on the range.
"Let's have dinner at home."
Couldn't we do this at home?
So this is Home on the Range.
"They make you feel at home."
"Home, home on the deranged."
"I love out of home cooking."
Nobody cooks at home anymore.
"Well, they DO say home-style."
This sure beats eating at home.
"Home cooking--what a concept!'
The sign does say home cooking.
"Home home-style on the range."
They specialize in home cooking
This sure beats cooking at home!
Just once, can't we eat at home?
At last, a real home-cooked meal.
"They advertised 'home cooking.',
The sign said home style cooking.
I feel right at home on the range.
I'm so glad we didn't eat at home.
"They make you feel at home here."
I prefer this to home cooked food.
"It has all the comforts of home."
I feel right at home on the range.
They call it Home Depot for diners.
You could have cooked this at home!
This is a little too close to home.
"I always enjoy a home cooked meal."
This is your idea of "Home cooking ?
"well, home cooking was advertised."
"Beats eating at home on the range."
We might as well have stayed at home.
Now this is what I call home cooking!
"Couldn't we have done this at home?"
Is a home-cooked meal too much to ask
"I really wish we had stayed at home."
Isn't this nicer than cooking at home?
They are known for their home cooking.
The sign did say "Home-Style Cooking".
"I hate these home-style meal places."
"We could have eaten at home tonight."
Home cooking is NOT what I expected...
"It's to make one feel right at home".
"It has a real home-cooked feel to it."
"Their specialty is home-style cooking"
The Groupon did say home style cooking.
Feeling right at home......on the range
After this, want to catch a home movie?
You call this a "home style" restaurant?
Home Home on the range? I want a divorce
"When did home cooking lose its luster?"
"Real home cooking at it's best I'd say"
Well, you wanted home-style cooking... .
They have the best home cooking in town.
There's nothing like a home-cooked meal.
"But the coupon did say - Home Cooking!"
"Well, The sign did say 'Home Cooking'".
"A New Concept in Home Cooking," indeed.
This does Not count as a Home cooked Meal
This is the home of the home-cooked meal.
This home cooking craze has gone too far!
'That's why they say it's 100% home made'
"I feel at home here. Home on the range."
They promise a "home-cooked" taste, honey.
They say its almost as good as home cooked
"This is what they mean by `home cooked`".
I just come for the home-cooking ambience!
You did say you wanted a home-cooked meal.
I hear the food is as good as home-cooked.
why did we go out for a home cooked meal.?
Everyone panhandles. at Home on The Range.
"There's nothing like a home-cooked meal."
Just like your good ol' home-cooked meals.
Cheap booze and pajamas are my home style.
"They seem to feature home-cooked dishes."
"Dear, couldn't we have done this at home?"
It's a Home on the Range dining experience.
So the ad was serious about n"Home Cooking"
"The website did say 'home style cooking.'"
They featured home cooking.................
"This is not the home cooked meal I meant!"
It was voted 'Most Authentic Home Cooking.'
They specialize in home-cooked comfort food.
"It is the next best thing to home cooking."
"Says here they're a home-style restaurant."
"This brings a new meaning to home cooking."
Doesn't this take "home style" a little far?
"They say this is better than home cooking."
Couldn't We've done this for cheaper at home
They are highly rated for their home cooking.
Let's get something we wouldn't make at home.
"This is your version of a home-cooked meal?"
Well, they do advertise real home cooking....
"Well, they did advertise home cooked meals."
"The prices! I'd rather be home on the range."
They really deliver on the 'home cooking' part
"Trust me, it's gonna taste like home cooked."
"No, you do it! I do all the cooking at home."
"I'm getting the 'Home on the Range' special."
The Times reviewer did say it was home cooking
"They take home cooking a little too seriously"
It's the next best thing to a home-cooked meal.
Nice, but I prefer dining at home on the range.
It's about time we had a nice, home-cooked meal.
Well you said you were tired of cooking at home.
Now, this is what I call good 'ole home cooking!
"How is this different from me cooking at home?"
This is carrying the home cooking theme too far.
Why couldn't we get a stove and do this at home?
Well, you said you were sick of cooking at home.
It's not me that wanted a good home cooked meal.
This gives a whole new meaning to "home cooking"
The cowboy said it's called "Home on the Range."
"Hon, do I hear you humming, "Home on the Range"?
"So this is what they call 'Home-style Cooking'?"
Finally, a real home-cooked meal at a restaurant!
When they say "home cooked " they really mean it!
"This is taking 'home cooking' a little too far."
So you literally meant "tired of cooking at home!"
They took make yourself at home to the next level.
Not the home cooking on the range I was expecting.
Why do we come here? You know I hate home cooking!
The sign did day home cooking but this is just nuts
"There's no cooking like home-style. Right, woman?"
"It's a nice way to ease you into cooking at home."
“The Home on the Range Special looks tempting.”
"So this is what they meant by a home-cooked meal."
You said you wanted a place with real home cooking.
You wanted to eat out. I wanted a home cooked meal.
What did you expect when the sign said Home Cooking?
Home-style cooking has taken on a whole new meaning.
I thought we went out to get away from home cooking.
At these prices, I'd just as soon we cooked at home.
Maybe we should eat at home until the next new trend.
I guess all we can have is homemade home-style fries.
"It says to eat here for home-cooking away from home."
"I understand that 'home cooking' is the latest trend."
You said you wanted a restaurant that does home cooking
"I didn't quite think 'home-style cooking' meant this."
And I booked us a ride home on something called uDrive.
For these prices, we could have stayed home and cooked.
"I think they're carrying this 'home cooking' too far."
I've heard of home on the range, but this is ridiculous.
"This 'home cooking' concept is getting out of control."
"Well, they did say "home-style cooking" in the review!"
"Honey, did you remember to turn the stove off at home?"
Coulda stayed at home and ate on a stove. Cachet my ass.
There's nothing like eating a good home cooked meal out.
"Well, what else would you expect from Café Home Depot?"
Home cooked, my ass. We're not home, and it's not cooked.
You're right, this really does remind me of home cooking.
“Is this what they mean by ‘home style cooking’?”
Well, they did advertise their specialty was home cooking.
"I think they've carried their home cooking idea too far."
Even though we're eating out, I feel at home on the range.
The ad said "home cooking" but this is not what I expected!
"I can't believe they made us print out our menus at home."
"These self serve restaurants make you feel right at home!"
"This is not what I was expecting by 'home-style' cooking."
I didn't imagine this when they said home style restaurant!
"Home cooking hibachi style... Who would've ever thunk it?"
This is not what I thought their ad meant by "home cooking."
"I'm thinking something we wouldn't typically make at home."
"They say it's the most authentic home-cooking in the city."
Now I know why the maitre d' was humming "Home on the Range"
"Cooking at home also means cleaning up at home. Lighten up!"
This chef is really means it when he promises "Home Cooking."
I didn't think they'd be so literal about 'home style cooking'
Hey, it's the next logical step after pack-your-own take home.
With these prices we could have stayed home and ruined dinner.
Not what I meant when I said I wanted the "At home" experience.
How about some less authentic home cooking next time we go out?
"Here's something you don't see everyday—home style cooking."
I'm sorry: I thought "Home on the Range" meant a Western-theme.
Now I know what the reviewer meant by "that home cooked flavor"
What did you expect? This is the "Home on the Range" restaurant
"At these prices, we should have just stayed home on the range."
I think their promise of home cooking is going a little too far.
Do you think they're taking this home-cooked meal thing too far?
"I think they're pushing this 'home-cooked meal' thing too hard."
You know it's about time we went out for a nice home cooked meal.
What can we charge for this self service/home cooking experience?
"Dear, this is the closest to home-cooking you are going to get."
When they said it was "home-style cooking", they weren't kidding.
I better not hear "I could have made that at home" when we leave.
We've come to the point where 'home-cooked meals' is an oxymoron.
"If I hear 'Home On The Range' just one more time, we're leaving."
When they said "Home Cooking" in that ad, they meant home cooking!

[...]


cluster 10:
===========
Oven for two?
"Oven ready."
Oven mitts $5.00?!
It's an oven in here.
Oven mitts are extra.
This table is an oven
Want to go Dutch oven?
Do you come here oven?
Did I leave the oven on?
So this is Oven-to-table.
Did we leave the oven on?
"Oven mitts are $2 extra."
You have a bun in the oven?
"It's a farm-to-oven place."
Let's put a bun in that oven!
"Absurd. No Brick Oven Pizza."
"I hope they left the oven on"
"Let's go dutch oven tonight."
"Wanna put a bun in the oven?"
I hope it's a self-clean oven!
Do you come to this place oven?
You've got a bun in which oven?
"It's like an oven right here."
Did you remember our oven mitts?
Did you remember the oven mitts?
I see you have a bun in the oven.
"Shall we put a bun in the oven?"
What? You have a bun in the oven?
Honey. 9 o'clock. Bun in the oven.
Hillary, our legs are in the oven.
"I told you oven-to-table was hot."
The oven-fried chicken sounds good.
Let's start with a bun in the oven.
"Shall we splurge and use the oven?"
the bun in the oven looks delightful
The oven comes with the meal, right?
Pricey, for a Bring Your Own Oven...
"Did we forget to turn off the oven?"
It's true, we have a bun in the oven.
" This is "convection oven" prices !"
What? I told you, it's Oven to Table.
I specifically requested a brick oven.
I wonder if it's a self cleaning oven.
My favorite is the oven fried chicken.
I think I'm ready to pre-heat the oven.
Don't put your elbows on the oven, dear.
I'm thinking the "Oven-Stuffer roaster".
Can I interest you in a bun in the oven?
I'm game for putting the bun in the oven.
Did you remember to bring the oven mitts?
What do you mean there's a bun in the oven?
"No wine tonight. There's a bun in the oven."
Did you remember to turn off the oven, honey?
"I can't keep my legs in the oven that long".
So this is as close as you'll get to an oven.
An oven table is the new hip thing... I think.
"Stove top stuffing or oven-browned potatoes?"
"We should start the oven now for my selection!"
"I wonder what the 'one in the oven' special is?
Yes, but here they don't store towels in the oven.
As a starter, I strongly recommend the oven mitts.
This whole 'oven to table' thing has gone too far.
"This reminds me, honey, did you leave the oven on?"
By the second date, she already had one in the oven.
"That reminds me, did you turn off the oven at home?"
If this date goes right, I get the girl and the oven.
"No wine tonight, honey. We've got a bun in the oven."
They are real innovators in the farm-to-oven movement>
"This isn't what I meant by putting a bun in the oven."
I'm glad we're partaking of the new farm-to-oven trend.
"When they say 'hot from the oven' they really mean it!"
These BYOO's (bring your own oven )should let you BYOB !
The Melting Pot, Where You Pay To Slave Over A Hot Oven.
"The roasted duck comes with two sides and an oven mitt."
Oven roasted turkey with the stove top stuffing for me...
Same ol', same ol'. Let's just put our heads in the oven.
The 'Lasagna Special" comes with complimentary oven mitts.
"I think we should order oven-baked instead of stove top."
Let's hope 'Loving from the Oven' is on tonight's Specials
I'll put something in the oven, you can have the stovetop.
I wonder why putting a bun in the oven is not on the menu.
I hope it's not BYOO because I forgot the oven-mitts at home.
If this is an oven, how come we can fit our legs underneath it?
Where is our waiter? Our oven preheated a good ten minutes ago.
“Is this your way of telling me there’s a bun in the oven?”
We'd better preheat the oven now if we want to see an early show.
If we go out for Italian tomorrow, we get to sit in a pizza oven.
Oven-tually you will have to chose something Martha, I'm Stoving.
Why don't you order while I open the oven and check on the kiddos?
Darling, Im so glad I don't have to slave over a hot oven tonight!
They should have named it the Lovin' Oven instead of the Hot Topic.
"Nothin' says lovin' like somethin' from the oven--The Management."
"I don't want anything that has to be in the oven for nine months."
When I was a kid my parents took me to the easy bake oven restaurant.
So, when you say "we have a bun in the oven"...you mean what, exactly?
I hope you didn't bring me here to tell me you have a bun in the oven.
"They come right here to your oven and burn the hell out of your steak."
You pretend to be so innocent and now you say you have a bun in the oven?
I'm sorry, but I don't think I'd feel comfortable putting a bun in our oven.
"Yes, this is a bad date, and no, you cannot stick your head into the oven."
"Martha I think I left the oven on." "David we still have 5 minutes on the timer."
"Honey, not sure if this is the best time to tell you but I got a bun in the oven."
There's no rush to order. This is an E-Z Bake oven and I brought a higher watt bulb.
Well if you expect your chocolate soufflé to be ready for dessert you better get it in the oven!
A'la-Burner prices are reasonable....but, there's a surcharge for the Fire Extinguisher and Oven!
"I think I'm going to go for the 4-burner special again. Are you okay with ordering from the oven?"

106 captions.


[...]




cluster 13:
===========
I'll cook tonight.
"I'll do the dishes."
You cook. I'll clean.
"I'll have the sushi."
"I'll pay if you cook."
"You cook, I'll clean."
"You cook, I'll clean."
"I'll have the D.I.FRY."
I'll have what he's making
"I''ll get the Jiffy Pop."
I'll have the steak tartar.
"Fine, I'll cook you wash."
I'll have what he's making.
I'll cook what he's cooking.
I'll have the steak tartare.
I'll have the home made eggs
"I'll make what he's having"
"I'll make what he's having"
"I'll have what he's having"
"I'll cook if you clean up."
"I'll have the roasted nuts."
I'll have the sushi deluxe...
You cook - I'll do the dishes
I'll cook if you wash dishes.
I'll pay if you do the dishes
I think I'll make mine to go.
I'll just have some of yours.
"I'll have what he's having."
I'll have what you're making.
I'll have what they're having.
"I think I'll get the lobster"
I'll have the 'selfie omelet'.
I'll have what they're having.
I think I'll have the sashimi.
I think I'll have the fahitas.
"I'll have what you're having."
I think I'll have the gazpacho.
"I 'll have what he is having."
I think I'll have the gazpacho.
"You cook, I'll do the dishes."
"I'll cook, you do the dishes."
"I'll have what you're making."
"I'll cook, you do the dishes."
"I'll have the duck a la range."
I'll take mine medium-rare, dear
"I'll have the Boeuf a l'Amana."
If you cook, I'll do the dishes.
If you cook, I'll do the dishes.
"I think I'll order the sashimi."
" I think I'll have the stuffing"
"I'll have the air conditioning."
I think I'll have the boiled egg.
I think I'll have the cold plate.
"I'll cook dinner tonight, dear."
I'll cook if you'll do the dishes?
"I think I'll have the leftovers."
"I think I'll have the TV dinner."
"I think I'll have the TV dinner."
I'll blanch what you're blanching.
I'll just go and kill the pig then.
" I think i'll order the Jiffy Pop"
I think I'll get the chicken stock.
"I think I"ll have the cold plate."
I'll cook if you'll do the dishes...
"If we hurry, we'll make the movie."
"I think I'll have the vichyssoise."
I’ll go with the leftovers buffet.
What's wrong? They'll do the dishes.
I think I'll have the steak tartare.
"I'll cook, but you're doing dishes!"
"Think I'll have some frozen lasagna"
"I'll cook yours if you'll cook mine!"
I'll start with the stove top stuffing.
I think I'll have the Hamburger Helper.
"I think I'll go with cold cuts tonight"
I think I'll have the Lobster Thermador.
I'll have the Hansel and Gretel special.
"Do you think we'll need to leave a tip?"
"I think I'll do the homemade apple pie."
"I think I'll have the Blue Pot Special."
"I think I'll order, 'Stovetop Stuffing'!"
"I'm hon and I'll be your server tonight."
I'll cook. You hold the fire extinguisher.
I'll warm up the leftover tuna casserole .
I think I'll have the stop, drop and roll.
I think I'll pass on ice cream for dessert.
"I'm Pete and I'll be your server tonight."
I'll clean up if you make my favorite dish.
"This time I'll cook and you send it back."
Well, in that case I'll get the vichyssoise


[...]


cluster 14:
===========
"Just like home."
Like homemade food.
"I feel like a salad."
Just like home cooking.
See any recipe you like?
Wow, just like homemade.
"I hope you like stuffing."
Hope you like home cooking.
I like the chilled gazpacho.
"Hope you like home cooking."
"I'd like a cold cut sandwich."
"I hope you like home cooking."
It's like a mid-western Hibachi
My mother had a table like this.
"What do you feel like cooking?"
"I hope you like Spagehetti-os."
"What do you feel like cooking?"
"I feel like a big-ticket meal."
"But I feel like something cold."
" Not like but exactly homemade. "
We have to stop meeting like this.
"Looks like homestyle Teppanyaki."
"I feel like ramen, how about you?"
Looks like more home cooking to me!!
It's like a Benihana American style.
I don't feel like cooking, lets bake.
It's like Benihana - for the midwest.
Gee, it's just like home on the range
"I can burn anything you'd like dear."
"Do you like peanut butter and jelly?"
"Would you like to hear the specials?"
My mother is really going to like you.
Suddenly I just feel like a big salad.
I wonder what the bathroom looks like.
"Looks like a great range of choices."
Funny, this reads more like a cookbook.
"Ever feel like you left something on?"
"Looks like they've got quite a range."
Would you like to split some Jiffy Pop?
"They say it's just like home cooking!"
Dear, what would you like to burn today?
"What do you feel like cooking tonight?"
I don't feel like cooking. Let's eat in.
It looks like everything's in our range.
I don't feel like cooking, let's go home.
Now we can prepare it the way we like it.
I'd like mine well done, how about yours?
So, what do you feel like cooking tonight?
"I really don't feel like cooking tonight"
"Let's see...what do I feel like burning?"
I said I didn't feel like cooking tonight.
So, what do you feel like cooking tonight?
"It's for Koreans who don't like barbecue."
The menu does say "Just like home cooking."
I was told this place tastes just like home
“So…what do you feel like making us?”
"It's just like being at home on our range."
I hear the food is just like "home cooking."
Just like dining at home and no cleaning up!
Looks like their lawsuit policy is ironclad.
I really do not feel like cooking tonight...
I'd like to put something on the back burner
They say the food here tastes like home made!
Lately I feel like we've lost our convection.
"The 5-course meal seems like a lot of work."
"It's like a cookout, only without the ants."
"Says here it tastes just like home cooking."
They said it was just like real home cooking.
It's pricey, but it tastes like home cooking.
"I thought you said you like home-style food."
How do you like our anniversary dinner so far?
Alright, I like it,but I'm not doing the dishes.
" They say it's just like mother used to make ".
It's like Korean barbecue but with mac & cheese.
Let's go home, I don't feel like cooking tonight.
I told you it would be just like my home cooking!
Going out to eat just seems like work these days.
It looks like we have a range of options tonight.
They say everything tastes like home-cooking here!
"I really don't feel like cooking. Let's go home."
They say the food here tastes just like home made.
"I don't feel like cooking tonight. Let's go home."
I feel like a fool ... but I feel like vichyssoise.
DIY Cooking for that "tastes like home" experience.
"Let's go home. I don't feel like cooking tonight."
"Let's go home. I don't feel like cooking tonight."
Don't you like it when I take you out of the house?
More crematoriums should have cafeterias like this.
I don't feel like cooking; lets go home and watch TV
Just what I like: home cooking at restaurant prices!
"Don't hate me but I feel like going for the sushi."
This is good—I really felt like being seen tonight.
I understand the food here is just like home cooking.
Does it smell like gasoline in here to you right now?
Let's go home; I don't feel like cooking out tonight.
I don't feel like cooking tonight, let's eat at home.
"'I don't feel like cooking, let's go out.' she says."
Let's eat at home tonight...I don't feel like cooking.
"No dear, pre-heating in nothing like hors d'oeuvres/"
But I thought you said that you didn't like my cooking.
"I'm sure you'll like it. It's just like home cooking."
"Looks like I got the back burners and slower service."
"Would you like to split the do-it-yourself Consommé?"
"There's nothing like good old-fashioned home cooking."
I like the ambience, but the food always comes out cold.
"Let's go home. I just don't feel like cooking tonight."
"I like it when I don't have to clean up after cooking."
I wanted to see what you looked like in front of a stove.
"I thought you said you didn't feel like cooking tonight."
Hey Hon, what do you feel like cooking for dinner tonight?
Looks like the purge got rid of all the waiters and cooks.
"Don't complain - the sign says 'just like home cooking'."
When they said "just like home made" they really meant it.
I would hate to think what the cocktail lounge looks like.
"Well, the review did say 'tastes like good home cookin' "
“I don’t feel like cooking tonight, let’s go home.”
How would you like me to make you a roast turkey, my treat?
I don't really feel like cooking tonight. How about a salad?
It says, "Guaranteed to be cooked just the way you like it".
I don't feel like cooking tonight. Let's go home for dinner.
"If you like it, you may want to use the one in our kitchen."
When they said "Just like home cooking," they weren't kidding.
I don't feel like cooking tonight. I'll have the frozen pizza.
I know you want Titanium, but I feel like a comfort-cast iron.
It's just like Japan except we don't have to sit on the floor.
Would you like to share the 20% off do your own dishes special?
"But you said 'you didn't feel like staying home and cooking' "
Ooh - it looks like they've updated their left-front selections.
Calling this "A Benihana-Like Experience" is a bit of a stretch.
“Let’s just order a sandwich. I don’t feel like cooking.”
Says here, 'Tastes just like home cooking!' ...there's a shocker.
I love it. It's like home cooking, but we don't have to clean up.
"I'm starting to feel like our lives are an exercise in futility."
Somehow it doesn't feel like dinner at home without our cell phones.
There nothing like fresh food off the stove, but this is ridiculous.
"Now we know why it said, 'tastes like home' on their front window."
"I'm glad we did this. I just didn't feel like not cooking tonight."
"Burnt meatloaf? Not everything works because its 'just like home.'"
Looks like we won't be sending any food back to the kitchen tonight.
"I don't really feel like cooking. Let's go home and order Chinese."
"For some reason I feel like having the 'Gaffers & Sattler Special'."
"I don't feel like cooking tonight. Let's just call out for a pizza."
"They say the food here taste just like you would have made it at home"

[...]

cluster 15:
===========
Nice try.
"Let's try the sushi."
Should we try a flambe'?
"I'll try the raw meat”
"Let's try the Chef's Special."
Elbows on the table? Just try it
Nice try, but I'll have the sushi.
I think I will try the "Left Overs"
Try not to burn the meal this time.
You should try his at home sometime!
All right Dianne, I'll try the fugu.
You should try this at home sometime!
I think I'll try table 2's leftovers.
Should we try their proctology clinic?
We should try cooking at home sometime.
Try the veal. It's the best in the city.
"I think I'll try the Calphalon tonight."
Next week, let's try the microwave place.
Think I'll try the Sylvia Plath omelette.
"I'm going to try the Stove Top Stuffing."
"Try not to order anything labor intensive."
"Try the stovetop stuffing,Its the specialty"
Next time let's try a place with a microwave.
"I said I wanted to try a BYOB, Not BYOP (Pot)"
I've been dying to try out this DIY restaurant.
This is such fun-- I'm going to try their PB&J.
I'm burnered out today. Lets try the casserole.
We should try that Refrigerator place next week.
"I think I'll try the deer-and-antelope special."
Try to avoid the "do-it-yourself" side of the menu.
Oooo,honey they have a cat, No I want to try dog....
I want to try something new, like cereal or fresh fruits.
"You win. Next time we go out, we'll try the sushi place."
I think I'll try putting my head in the oven this evening...
Why don't you try the house white, and I'll try the house red.
"You should try ordering something you've never burned before."
So, this is the hot new restaurant you've been wanting to try...
Should we try the wok and chopsticks or the saute pan and whisk?
Next time you want "home style" try checking some reviews first.
So we pay extra to cook our own food? We should try this at home!
Next week we should try L'Auberge Microwave. The service is faster.
"Let's try to enjoy our dinner and put our problems on the back burner".
...And next Friday I thought we could try that new place, "Le Dishwashier".
"Oh wow! Let's try their recipe for Purple-gilled Laccaria mushroom gnochi!"
"I'll give them one last try. Last time I was in the scallops were overcooked."
I feel confident. Let's try one of those "most challenging" recipies from the Iron Chef column.
You know, every time we come here I swear I'll try something new but then I get here and I just can't resist the Hamburger Helper.
I'll try the stir fry and hope to minimize the third degree burns. I still have burn scars from your last attempt at making an omelette.
(The sign on the window should read, " Home Cooked Meals". I think we should try the Southern Fried Chicken and mashed potatoes recipe. What do you think, dear?

49 captions.


cluster 16:
===========
Chef's night out.
I call sous chef.
Chef Boy Are We !
"I know the chef."
The chef walked out
Who is Chef Yer Self?
"The chef went union."
Chef's revenge on foodies
Chef must have quit again.
The chef salad looks good.
No, YOU'LL be the sous chef.
“Chef’s choice again?”
'I hear the chef is amazing.'
How do I compliment the chef?
They all say "Chef's choice."
"What is the chef's special?"
The chef suggests medium rare.
"I hear the chef is fabulous!"
"They say he's a chef's chef."
"Everyone here loves the chef."
I'm thinking the chef's special
"The chef recommends non-stick"
The Chef's Special sounds good.
Will the chef's strike ever end?
The chef recommends an oven mit.
the paper gave the chef 5 stars.
"Your own personal chef my ass!"
This redefines the Chef's table.
The chef got tired of complaints.
"I hear the chef was burned out."
"The chef's special sounds good."
"I'm getting the chef's special."
I've heard the chef is incredible.
I told you the chef was on holiday.
"Chef Selection List...impressive!"
The kitchen's full of Chef's Tables.
"Irony Chef"? Are you doing dessert?
Apparently the chef comes all carte.
"Apparently, the chef is off today."
I hear the chef's range is terrific.
I heard the chef here is incredible.
The Yelp reviews rave about the chef.
As a first course, lets order a chef.
Glad we watched Iron Chef last night.
"Have you spotted our sous chef yet?"
Chef recommends "Cooking for Dummies".
"It says it's a chef-free restaurant."
"There's a 10% surcharge for the chef"
I hear the chef is passive aggressive.
I wonder what the chef's table is like.
Well, I did request the chef's table...
What a find, no tipping, no chef visits.
At these prices I expected a stove chef.
The Executive Chef specializes in D.I.Y.
My specialty is complaining to the Chef!
The pasta a la Chef Boyardee looks good.
"I've heard great things about the chef."
I've read great reviews of the chef here.
My first wife is a Michelin starred chef.
THIS is what they mean by "chef's table"?
Boy, this chef sure knows how to delegate.
"They don't seem to have a Chef's Special."
"The reviews did say we will love the chef."
It's owned by an Hibachi chef from Alabama .
Next time we're sitting at the chef's table.
"Do you want to be tonight's Celebrity Chef?"
This gives "chef's table" a whole new meaning.
Chef doesn't think much of my culinary skills.
Boy ...this chef seems like a real prima donna
The chef recommends the Chateaubriand tonight.
I hear this is where all the best chef's eat...
That's the last time I'll ever insult the chef.
We are waiting for the food. Where is the chef?
Chef's tables just aren't what they used to be.
"Hmmm... 'the chef is open to suggestions' ..."
They don't have a chef, just a produce manager.
So, this is how the chef gets away on vacation.
Before dessert, I'd like to compliment the chef.
These new Chef Les restaurants are all the rage.
"Yes, and I happen to know the chef personally!"
I hear the chef comes straight from our kitchen.
"Hmmm, I thought Chef You was actually a person."
Ahh, I'm getting the chef special: Ramen Noodles!
Apparently the chef doesn't like the cooking part
Yup, the chef's suggestions is a range of choices.
I heard the restaurant's chef has his own TV show!
I'd give the chef here five stars for pure idleness.
The chef recommends we keep our elbows off the table
I can't believe we paid extra for the "chef's table."
"For an extra $50, we can add the chef to our order."
I hear the chef's wife is going to ask for a divorce.
I'm definitely ordering the chef's cold plate special.
I think I'll order the table-side chef at $75 an hour.
"Jim, here, a support chef assists with boiled water."
I saw the chef use the bathroom and not wash his hands
Shall we order the Novice, the Top Chef, or the Poseur?
The chef here is highly recommended by me, myself and I.
Darling, let’s come back when the Chef strike is over.
Until the chef strike is settled I'm sticking to salads.
Apparently, they've had a lot of problems keeping a Chef.
"Can you ask the chef not to overcook my steak this time?"
It says here the chef most recently worked at Votre Maison.
"This is the LAST time we're reserving the 'Chef's Table'."
"If it's good, let's send our compliments to the Sous chef."
This chef has inspired me to make some changes at the office.
The chef specializes in preconstructing some of the classics.
"Was that the chef out front yelling about how unfair life is?"
Well why did you bring us here if you heard the chef was awful?
How can I complain about the food at this place if I'm the chef?
"I always feel insincere when I give my compliments to the chef."
I hear the chef is is not in the kitchen as much as he used to be.
"And it's the chef's night off tonight, so we get a 20% discount."
I hear their best chef pays the owners for the opportunity to cook.
"I've had the 'chef's surprise' before, so it won't be a surprise."
The chef gets paid for doing half a job - just like the cartoonist!
"I don't know why I keep on coming back? I hate the chef's cooking!"
Perhaps, we should get recommendations from the chef before we order.
Did you see the fine print: "No chef. We pass the savings on to you!"
When every man's his own chef, the customer has no one to complain to.
I heard the Chef was once thought of as a perfectionist in the kitchen!
"I told you that chef would give the owner attitude one too many times."
How long must we wait for the teppanyaki chef to arrive to cook our dinner?
"Let's not argue about this, which one of us is the 'Designated Chef' for tonight?"
They say what the chef lacks in execution he more than makes up for in inventiveness.
If you ask me, the chef is overly concerned about his liability for undercooked foods.
"This All you can cook restaurant is very affordable but I've heard the Chef is terrible"
To be fair, François warned us that frequent criticism of the chef would have consequences.
When they said it was Chef’s night off, I assumed they just meant that we’d have a different chef …
Unless we can persuade the chef to come to our table, we will have to opt for a la carte and boil up some pasta...
This seems a bit of an extreme cost cutting measure - firing the Chef??? Or maybe we are part of a grand experiment? Or a new reality show?
"Before the starters I see we have a little personal note from the Chef ." He says,"if you think you can do better have at it."Bon Appetite.
Sorry Sally, the 21st Century InductionTable was already taken. And they don't have the vintage pink Kelvinator. There was an ugly old Magic Chef, but I settled for this.
So, I read in the Times review that the chef here has a big ol' dick. Honey, did you hear my joke? I was saying that the chef here -- which, in this case, is me -- has a super big dick. Honey, why are you crying?
... then the maitre d’, headwaiter, chef, and busboy all defecate, ejaculate, urinate, and expectorate into a sauce pan, mix it all together, and cook it on these handy stove tops. And the name of this restaurant? “The One Percent!”

134 captions.


cluster 17:
===========
Farm to Table
Farm to Stove Table
"It's farm-to-stove."
I prefer farm to table
"I prefer farm to table."
Farm to table skips a step.
So this is kitchen to table
“Hottest table in town.”
Did you say "farm-to-table"?
" Is Farm the Special here "
It's the new "farm-to-table."
At least the table is toasty.
"It's the new farm-to-table."
Farm to table is so last year.
"Elbows off the table, Madge."
I really prefer Farm-to-Table.
It's farm to table, non- stop.
Farm to table has gone too far.
"Another frack-to-table joint?"
This must be Farm to Stove Top.
Frankly, I prefer Farm to Table.
"Is this part of farm-to-table?"
"This place is real farm to table"
Nix that kitchen-to-table movement
They're calling it 'Stove-as-Table.'
I think I left the table on at home.
"Table-side beef flambe sounds good."
“Keep your elbows off the table.”
The Farm-to-Stove movement is genius!
Did you remember to preheat the table?
It's the new movement, range to table.
Did you remember to turn the table off?
Farm to stove is where I draw the line.
So, this is what farm to kitchen means?
I wish we had gotten the sous vide table
"It's a FARM to RANGE dining experience"
This place takes farm-to-table seriously.
I was just getting used to farm to table.
I agree, "farm to table" is so last year.
I told you "warm"-to-table wasn't a typo.
It says to keep your elbows off the table.
Remind me to keep my elbows off the table.
“Apparently we’re serving Table 12.”
"See that guy with his elbow on the table?"
Remember to keep your elbows off the table.
"Well, at least our table is not wobbly..."
Farm to table just involves too many steps.
Its called farm-to-table for good reason….
I'm dubious about this farm to stove cuisine
"Wow. It IS impossible to get a table here."
Farm to table just seems so inefficient, no?
It's Farm-to-table. Set your burner on HIGH.
I liked it better when it was farm-to-table.
Darn, we were first supposed to farm our food.
Do you mind if we switch to an electric table?
For my money, this beats farm to table anyday.
I thought you said it was farm to table dining?
This farm-to-stovetop thing really burns me up!
"'Organic, savory, farm-to-table... Jiffy Pop'"
This farm to table trend has gotten out of hand.
I hate these new "Kitchen to Table" restaurants.
"Farm-to-table left me with too many questions."
It's the new farm-to-rangetop-to-mouth movement.
"They're bridging the gap between farm to table."
Their serious about keeping elbows off the table.
"You could have asked for the five-burner table."
So then, technically, this isn't "farm-to-table."
I thought it was farm-to-table, not farm-to-stove.
"Just remember to keep your elbows off the table."
I'll be glad when this stove-to-table fad is over.
I miss the days when food was prepared table side.
"Farm-to-table restaurants...those were the days."
It's like farm to table. Except without the table.
This “Farm to Table” actually skips the kitchen.
It's the latest thing in the farm-to-table movement.
This whole stove to table movement has gone too far.
I'm not so sure about this "Farm to Stove" movement.
I'm a big fan of this new stove to table initiative.
My mom always said not to put my elbows on the table.
"If you can't stand the heat get away from the table."
"I'm at a loss with this new 'home to table' concept."
Looks like they're also doing the farm-to-stove thing.
"Oh, that smell. The table behind us is on self clean."
They really take no elbows on the table here seriously.
It's farm-to-table, but they've cut out the middle man.
I'm not sure Farm to Stove dining is going to catch on.
You think liver and onions would bother the next table?
I think they've taken this range-to-table thing too far
I'll remember to keep my elbows off the table this time.
They say that farm-to-stove is going to be all the rage.
I'm not very hungry. Want to move to a two-burner table?
Ohhh! You said get the reservation on open table... Oops.
I've heard about "farm to table," but this is a bit much.
Apparently you need a reservation to sit at a prep table.
"Persons using this table are more likely to suffer burns."
It's called farm to stove-top, apparently it's all the rage.
"Talk about farm to table. They've eliminated the middleman."
I'm not sure how, but this removes a step from farm to table.
"It said it was "farm to table top", not "farm to stove top".
"Who knew this farm-to-stovetop thing would really take off?"
"I'm re-thinking my interest in their farm-to-table location."
I think the restaurant-to-table movement is really catching on.
Perhaps we should have stopped after the "farm to table" craze.
"When they said, 'Cooked at your table,' they weren't kidding. "
I know they said a table by the kitchen, but this is ridiculous.

[...]


cluster 18:
===========
"It's gas."
Gas surtax?
"I have gas."
"I smell gas."
Gas or electric?
gas or electric?
Gas or electric?
Do you smell gas?
Do you smell gas?
Do you smell gas?
Gas or electric ?
Do you smell gas?
"Do you smell gas?"
"Is the gas local?"
We always get gas !
" Do you smell gas?"
Gas or electric menu?
Let's just turn on the gas.
Now you're cooking with gas.
You look lovely by gas light.
Do you prefer gas or electric?
It says here that gas is extra.
"This place always gives me gas"
I think we should upgrade to gas
"Do you prefer gas or electric?"
I think this meal will give me gas.
You know I prefer cooking with gas.
" I got gas last time we were here "
This restaurant always gives me gas.
"We pump our own gas, and now this,"
“This place always gives us gas.”
It all started with self-service gas.
"This restaurant always gives me gas."
Are you going gas or electric tonight?
Why does this place always give us gas?
It all began with pumping your own gas.
This used to be the Gas-Stove District.
This natural gas prices are outrageous!
Do you think the gas is locally sourced?
At these prices, I'd expect gas burners.
Is the gas they use natural or fracked ?
Gas is included in the price of the meal.
"I hear that the gas is fracked locally."
"The special is Nova Scotia natural gas."
"I was told if you eat here, you get gas."
"Convenient, sure...but I still prefer gas."
I smell gas. I think the pilot light is out.
"They have a ten percent surcharge for gas!"
Well, you always said you wanted a gas stove.
Gosh, 1st it was 'self service' gas stations!
I cant decide if I want the gas or electric..
Modern Gas Cooker gives this place five stars.
"Now I understand why you asked if I had gas."
I thought the hyphen in gas-tropub was a typo.
"I like it better here, because they use gas."
Do you see how much they are charging for gas?
Don't skimp, dear. They're paying the gas bill.
At least here we don't have to pay the gas bill
"I hear the gas is organic and humanely drilled."
The ranges are electric but, this place is a gas.
"We never should have started pumping our own gas."
I'm so glad we finally got a gas range reservation!
Quick! If we order before seven, the gas is included
None of the gas stove tables were available tonight.
First it was pumping gas, then it was checkout lines.
They have both, but when I eat here I usually get gas.
This place was so much better when they had gas ranges.
"There is a separate charge for the amount of gas used."
"Well if it's gas I'm going with the halibut. But electric?"
They say the gas here is natural, fresh and locally fracked.
Are you playing footsy with me or did I just turn on the gas?
"The Whirlpool bistro on 52nd has locally sourced natural gas"
“I much prefer gas restaurants over electric restaurants.”
"When you said you had gas at this restaurant, I had no idea…"
Maybe we should cook with a vintage 1972 West Virginia shale gas.
First it was pump your own gas, then those self checkout things. . .
"And I thought it was bad when they started making us pump our own gas!"
"This electric stove is okay, but the restaurant next door gives me gas."
"A ten percent surcharge for gas? ... we're moving to an all-electric table."
" Hmmmm, cook your own meal. But a gas surcharge on top of the meal cost, I think not."
This is the last time you bring me to a stove restaurant that doesn't have a gas section.
The new service model provides a more interactive experience for the user, and the interface is customizable in gas or electric.
Do you prefer gas, conduction, or electric? Because I think we are in the wrong section. Wow, do you remember smoking sections?! So if the judge/chef likes it,is it free?

83 captions.


cluster 19:
===========
Where's our food?
Did you bring the Wok?
Did you bring the food?
BYOP: Bring Your Own Pans
Did you bring the aprons?
Did you remember the PAM?
"It's not about the food."
"Did you bring the paprika?"
It says recipes are extra...
"Did you bring your recipes?"
I thought tips were included?
Did you bring the big skillet?
"It's Bring Your Own Skillet."
"I thought you were a foodie."
Did you bring a pan? It's BYOP.
"Honey, did you bring our pot?"
It says utilities are included.
"Did you bring the calf testes?"
It says, "Potholders are extra."
Wait, did you bring the skillet?
We should bring your mother here.
We should bring your mother here.
Damn, we forgot to bring our pan!
And I thought that fondue was 2DIY
I thought we'd go out for a change.
Did you remember to bring the wine?
Did you remember to bring my apron?
"Says 10% off if we do the dishes."
Honey, it says right here B.Y.O.P.!
I thought you wanted the night off!
"It says preparation time can vary."
did you remember to bring the chicken?
It says no refund if you burn the food
I never thought of fast food this way.
It says"..plus tax and meter reading".
Did you remember to bring the paprika?
Probably good we didn't bring the kids.
" And I thought Self Check-Out was bad"
I thought we were going to a sushi bar!
And I thought fondue was a lot of work.
I hate these "bring your own food" places
"It says here that gratuity is included."
For Starters it says go grocery shopping.
My mother says the food here is excellent.
It says here that service is not included.
It says we have to do our own dishes, too.
I thought you said we could go get some pot.
" ONE-STOP STOVETOP"- The name says it all."
" I thought it'd be easier than a pre-nupt."
And I thought Benihana was a bridge too far.
"Oooooh. I thought he said 'Walk this way.'"
My mother says everything is overcooked here.
Next time, we really ought to bring the kids.
It says here we have to pick up the food, too.
"It says you have to bring your own kumquats."
Yelp says it's the hottest restaurant in town.
Wait. We're supposed to bring our own food too!
It's so romantic when they bring the accordion.
"It says 'to please pre-heat before ordering'."
I thought it was the cows that were range-fed !
Oh, no, I forgot to bring the porcini mushrooms.
"I thought you said we were going to a cookout."
The fine print says- "some assembly required..."
Did you happen to bring any extra soup with you?
I thought "The Amanian" was an ethnic restaurant.
So this is what they meant by bring your own pot.
It says 'Fast food like you've never had before.'
"It just says to go to the supermarket next door."
"It says we get to make our own carbon footprint."
"It says there's an extra charge for pot holders."
Next month they're introducing Bring Your Own Food.
It says here that all their food is cooked locally.
Did you remember to bring my favorite sauté spoon?
At least it's better than BYOF - bring your own food
It says here we need to bring our own groceries too.
"It says that Today's Special includes electricity."
I knew it was a Bring Your Own but not everything --
Did you forget to bring the refrigerator once again?
Did you bring a coat? I'd like ice cream for dessert.
I thought we'd agreed to stop talking about marriage.
I thought it would be so nice to get out of the house
I didn't bring the skillet. Did you bring the skillet?
It's the latest restaurant trend: Bring Your Own Food.
Never thought I'd miss cosy Japanese restaurants, dear.
Step 1 says to turn on the stove. Show it a little leg.
It says we're covered under their worker's comp policy.
"It says the theme is a fusion of feminism and sarcasm."
I thought you told me this was a small grate restaurant.
Why did we come back here? The food is always overcooked.
It says their motto is, "When you're here, you're staff."

[...]

cluster 20:
===========
order cold cuts.
Let's order out.
Let's order Sushi
Let's order salad.
Let's order sushi!
Made to order, eh?
"How do we order ? "
Let's order take out.
Let's order take out.
Let's just order out.
"Let's order the sushi
Don't order the sushi.
Don't order the salad.
I'll order --you cook.
Just don't order beans.
"Don't order ice cream"
Let's order out tonight.
"May I cook your order?"
I say we order the sushi.
Let's order the Gazpacho.
Don't order the wild boar.
Would you rather order in?
" Cooked to whose order ?"
Let's order the sushi to go.
Can I take your order, dear?
"Is it too late to order in?"
Please don't order the turkey.
"I wouldn't order the flambé."
We should probably order salad.
"Don't order the Baked Alaska."
"I might have to order cereal."
May I take your order, darling?
What pan are you going to order?
"What say we order out tonight?"
They won't let you order a salad.
"Let's pre-heat before we order."
Before we order, pre-heat to 350.
"Let's just order a refrigerator."
"How many burners shall we order?"
Let's order the stove top stuffing.
I think I'm ready to take my order.
"Order a bib for the bananas flambé"
"I think I will order the TV dinner."
What happens if I order the ice cream?
Why in the world would you order Sushi?
"I'll be right back to take our order."
"Order simple as I'm doing the cooking."
Whatever you do, don't order the flambe.
"We'd better order some of the Easy-Off."
"Maybe we should order something simple."
"I think we should order something cold!"
"Do I order your version or my mother's?"
You can order grilled cheese or Jiffy Pop.
"Please don't order the slow-cooked lamb!"
I think a sauté of some type is in order.
I'll order the aquarium for bouillabaisse.
I have a burning desire to order the steak.
Should we start with an order of Jiffy Pop?
Who order these pizzas without any topping?
Don't order anything that needs to marinate.
Honey, didn't you order the six ring burner?
"I'm afraid to order the stove top stuffing."
what homecooked meal should we order tonight?
Order something that involves little cleanup.
Should we order take out or shall we stay in?
"Cooked to order "has become "order to cook."
Why don't we split an order of celery sticks?
"Does it kill the purpose if I order a salad?"
“You take my order then I’ll take yours”
If you boil the water, I will order the Ramen.
"It's time to order when our menus catch fire."
"For dessert, let's order the no-bake cookies."
"If we order takeout can we use our own stove?"
"Good news, my dear. We'll be cooked to order."
After you read the owner's manual, we can order.
We'll be here all night if you order the turkey.
Let's order drinks before you watch me turn it on
"Would you be offended if I order steak tartare?"
I had a much different concept of "made to order."
"Order fast. The menus tend to burst into flames."
Oooo, let's order the flour, eggs, and canola oil.
If you order the cod, turn it half-way thru baking
Should we order now, or put it on the back burner?
"I guess do it yourself is the new made-to-order."
Do you think we should change the order to dine-in?
Don't order the veal, I didn't bring any parmigian.
" Order whatever you want, but make enough for two."
"Darn, it's too loud in here to order the soufflé."
I'd order the fish but it's always over cooked here.
"I wouldn't order the steak. I tend to overcook it."
Here it's not cooked to order, it's ordered to cook!
I'm not in the mood for cooking. Let's order sashimi.
Watch what you order -- I've been burned here before!
"Order something pre cooked dear, we're running late."
I'm probably going to order one of my own specialties.
We don't have much time...let's order the "Leftovers"!
" Let's order six eggs, four bacon and one frying pan."
"Maybe, for once, you won't be sending your order back."
I will take your order first and then you can take mine.
Order the tiramisu , I'd put my hand in the fire for that
Just order everything underdone, I'll take it from there.
I always factor in my dry cleaning bill when I order here.
"Let's order the eggs, the movie starts in seven minutes."
If we want to make the movie, we'd better order the ramen.
Let's order something hot to eat, otherwise I am very cold
If you order the sponge for dessert, I'll get the cleanser.
"Shall we break up before we order or after the bill comes?"
"What kind of grapes do you want to order from the wine list?"
I'd order the poached salmon but I have no idea how to make it.
Do you have a couple hours? I think I want to order the risotto.
"If you order the special, they let you do the washing up, too."
I say you order a salad for yourself and prepare a steak for me.
"If you're getting the ground beef, I'll order the hamburger helper"
Can we put past relationships on the back burner until after we order?
Tappan restaurant, tapas restaurant...what does it matter? Just order!
Shall I order the sea bass with a Le Cruset or Emeril Lagasse skillet?
I think I'm going to get the "Work ran late, let's just order take-out."
Be careful if you order the crabs...they've figured out what we're up to.
You've got the bacon and eggs in your purse; should we order a frying pan?
Why don't you order the ingredients, I'll order cookware, and we can share?
"It just seems like so much effort. I'm going to order a bowl of Corn Flakes."
"Why don't we share? You order the beef Wellington, and I'll get a nice salad."
"How about you order the sauce-pan and I'll order the skillet and we can share?"
" It's based on " You can do it. We can help. " We order just the ingredients. "
Should we order the one burner blue plate special or the two burner meal d'jour?
Last time we ate here my suit caught fire. Let's not order the flambe this time.
Let's make sure not to order too much, I don't wanna be up all night doing dishes.
Some places let you order from an iPad. I guess this is taking it to the next level.
"Hurry up and order. I'm getting the mousse for dessert and have to go out back to gather the eggs."

129 captions.


Top terms per cluster:
Cluster 1: hot plate stove date just special really slaving going ll
Cluster 2: menu says range think honey order cook don just cookbook
Cluster 3: cooking tonight ll homestyle dinner said dear meant food self
Cluster 4: salad just ll think going bar order want good cobb
Cluster 5: free range chicken think meant thought menu ll looks restaurant
Cluster 6: don know dishes think care tip leave make tell wash
Cluster 7: cook didn said food know want home tonight honey like
Cluster 8: microwave section table let better place time think sit menu
Cluster 9: home cooking cooked style range meal say did stayed feel
Cluster 10: oven bun mitts did table come hope legs let putting
Cluster 11: dishes good restaurant tip let sushi just think burner time
Cluster 12: place home hear new range town heard sushi street hot
Cluster 13: ll think cook dishes having just make making clean wash
Cluster 14: like feel cooking home just don tonight let looks tastes
Cluster 15: try ll let time home think place week sushi new
Cluster 16: chef hear table special heard recommends sous order ve boy
Cluster 17: table farm elbows stove movement kitchen new really thing remember
Cluster 18: gas electric smell gives prefer natural place locally restaurant stove
Cluster 19: bring says thought food did remember kitchen mother said extra
Cluster 20: order let don ll sushi cooked salad just burner think

and finally, the all-caps submissions (because they rule):
----------------------------------------------------------
BYOB!
BYOFBPIUCPSBWKEFCGLASWD
"WHY DON'T WE JUST ORDER A SANDWICH"
BYOF?
PREHEAT TO 350 DEGREES BEFORE ORDERING.
"IT WOULD HAVE BEEN CHEAPER AND A LOT MORE FUN TO HAVE CELEBRATED OUR ANNIVERSARY AT HOME!"
I HEAR I MAKE A TERRIFIC RISOTTO HERE.
U. B. CHEF'S COOK 'N DINE
WHEN YOU SAID THE FOOD WOULD BE COOKED AT OUR TABLE, I WAS THINKING BENIHANA.
" B.Y.O.F. ?"
BYOP?
NOW I KNOW WHY THEY ADVERTISED THE LOWEST PRICED MEALS IN TOWN !
```

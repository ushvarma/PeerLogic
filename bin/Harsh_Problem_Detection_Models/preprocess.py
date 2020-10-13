# generic function to preprocess reviews:
import re
from spellchecker import SpellChecker

def preprocess_reviews(reviews):
  code = ("expertiza","travser","questionnare","realestatemodel","getrubricarray","questarray","curquestionnaire","iquiry","questionnarecontroller","vmquestionresponse","capybaraerror","interfce","reportstrategy","visiualisation","potentialbuyer","sepc","nilclass","supplementaryreviewquestionnaire","pannernode","realestatecompany","codeclimate","househunter","offscreencanvas","herokuapp","webaudio","nomethoderror","constantsourcenode","quesparams","assignmentparticipant","bluetoothdiscoverysession","webidl","offscreenrenderingcontext","offscreencanvasrenderingcontext","oscillatornode","rpec","gemfile","rubocop","requesteduser","travisci","lowfi","bacui","importfilecontroller","metareview","oodd","assignmentteam","selfreview","amazonaws","agenttourlist","signuptopic","functionalites","rubymine","setvaluecurveattime","gradinghistory","simplecov","bluetoothadapter","issuecomment","tourscontroller","metareviews","umls","webbluetooth","scenairos","getrole","rspecs","agenttest","potentialbuyercontroller","customeroptions","standarderror","nameurl","assignmentcontroller","addcreatetopicsection","createconstantsource","baseaudio","reviewbid","reviewbidcontroller","rscore","assignmentscontroller","adminpanel","testfolder","potentialbuyers","memebershave","stereocontext","droptopicdeadline","getphoto","applicationcontroller","searchcontroller","responsecontroller","generatereport","hasallprivilegesof","htmlcanvascontextrendering","webglrenderingcontext","reviewresponsemap","timezonepref","gradescontroller","gemlock","appendcreatetopicsection","offscreencanvascontextrendering","deletedtour","mdfile","getlocal","parsererror","reviewmapping","oodesign","ooddprogram","coverrange","usinggit","gitgit","herokuwould","navebar","numericality","repositoryhttps","edgecases","expertzia","metareviewer","sqlexception","experitza","gdrive","assignmentquestionnaire","questionnairecontroller","setcurrentvalueattime","uncaughtthrowerror","customerbookingscontroller","runningbundle","createhouseinformation","exertiza","staticpagescontroller","createtopic","addtopic","tourmanagement","functionalties","devcenter","googleuser","applicationrecord","factorybot","ereadme","inquirydetails","funtionalities","existassignment","modelsand","baseaudiocontext","constantsourceoptions","foreignmodel","bookmarkratingresponsemap","bookmarkratingquestionnaire","degin","audioparam","signupsheetcontroller","screencase","participantsuper","setposition","setorientation","waveshapernode","biquadfilternode","betahttps","gitusing","isrealtor","ishousehunter","addquestionnairetablerow","popupcontroller","hasmany","hasone","debugbranch","userscontroller","userr","heatgrid","architcture","flowchats","interfact"
  )
  student = ("kunalnarangtheone","ychen","stereopannernode","swivl","ibwsfrvjmiytql","slwhv","iucqq","sidekiq","yzhu","nilaykapadia","jasminewang","bebacc","skannan","rustfmt","ocde","drupadhy","ajain","amody","upadhyaydevang","henlo","txmwju","kqbvycku","bdxxa","rxsun","bmyvjy","rommsw","travisbuddy","hhharden","appveyor","rahulsethi","rshakya","ziwiwww","nikitaparanjape","hounse","tourid","probablty","myaccounts","nainly","flazzle","folls","dhamang","dfef","afbc","eqsy","impliescode","jwarren","dodn","ferjm","jisx","coulhasdn","cbbdf","partipant","jwboykin","amogh","agnihotri","fdea","rbit","rbdoes","pronciple","sbasnet","kvtmnznc","ppvasude","ceec","edabe","namig","pptn","explainationit","urswyvyc"
  )
  spell = SpellChecker()
  for i in range(len(reviews)):
    reviews[i] = re.sub(r'[^a-zA-Z0-9\s]',' ',reviews[i]) # Removing special character
    reviews[i] = re.sub('\'',' ',reviews[i]) # Removing quotes
    reviews[i] = re.sub('\d',' ',reviews[i]) # Replacing digits by space
    reviews[i] = re.sub(r'\s+[a-z][\s$]', ' ',reviews[i]) # Removing single characters and spaces alongside
    reviews[i] = re.sub(r'\s+', ' ',reviews[i]) # Replacing more than one space with a single space
    if 'www.' in reviews[i] or 'http:' in reviews[i] or 'https:' in reviews[i] or '.com' in reviews[i]:
          reviews[i] = re.sub(r"([^ ]+(?<=\.[a-z]{3}))", "<url>", reviews[i])
    reviews[i] = reviews[i].lower()
    for word in reviews[i].split():
      if word in code:
        reviews[i] = reviews[i].replace(word,"code")
      elif word in student:
        reviews[i] = reviews[i].replace(word,"student")  
      elif(bool(spell.unknown([word]))):
        recommended = spell.correction(word)
        print(recommended)
        reviews[i] = reviews[i].replace(word,recommended)  